import json
import os.path
from pathlib import Path

import gdown
import pytube
from fastapi import FastAPI, logger
from faster_whisper import WhisperModel
from pydub import AudioSegment
import time
import requests
from threading import Thread



app = FastAPI()

# --------------------
# logic
# --------------------


def download_drive_file(url, output_path):
    gdown.download(url, output_path, fuzzy=True)


def extract_audio_from_video(video_file_path, audio_file_path):
    audio_segment = AudioSegment.from_file(video_file_path)
    print(audio_segment.duration_seconds)
    audio_segment.export(audio_file_path, format="mp3")


def get_highest_video_resolution(yt):
    video = yt.streams.filter(file_extension='mp4', resolution='1080p').first()
    if video is None:
        video = yt.streams.filter(file_extension='mp4').filter(
            resolution=lambda res: int(res.split('p')[0]) <= 1080).order_by('resolution').desc().first()
    return video


def download_youtube_video(url, output_path):
    start_time = time.time()
    yt = pytube.YouTube(url)
    video = yt.streams.filter(file_extension='mp4').get_highest_resolution()
    high_res_video = get_highest_video_resolution(yt)
    logger.logger.info(high_res_video)
    logger.logger.info(f"Video size: {high_res_video.filesize / (1024 * 1024)} MB.")
    video.download(filename=output_path)
    download_time = time.time() - start_time
    logger.logger.info(f"Download time: {download_time:.2f} seconds.")


model = WhisperModel("medium", device="cpu", compute_type="int8")
Path.mkdir(Path(os.path.join("records")), exist_ok=True)


def transcribe(audio_file):
    segments, info = model.transcribe(audio_file, word_timestamps=True)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    formatted_segments = []

    for segment in segments:
        formatted_segment = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        }
        formatted_words = []
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        for word in segment.words:
            formatted_word = {
                "start": word.start,
                "end": word.end,
                "word": word.word
            }
            formatted_words.append(formatted_word)

        formatted_segment["words"] = formatted_words
        formatted_segments.append(formatted_segment)

    print(formatted_segments)
    return formatted_segments


def save_data(file_path, data):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)


def segments_to_text(segments):
    text = ""
    for segment in segments:
        print(segment)
        text += segment["text"] + " "
    return text


def convert_seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{minutes:02}:{secs:05.2f}"


def segments_to_timestamped_text(video_segments):
    texts = []
    for segment in video_segments:
        start_str = convert_seconds_to_timestamp(segment["start"])
        end_str = convert_seconds_to_timestamp(segment["end"])
        text = segment["text"]
        result = f"({start_str} - {end_str}): {text}"
        texts.append(result)
    result = "\n".join(texts)
    return result


def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)


def read_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text


def run_google_drive_workflow(file_id, video_path):
    g_drive_url = f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link"
    download_drive_file(g_drive_url, video_path)


def run_youtube_workflow(video_id, video_path):
    yt_url = f"https://www.youtube.com/watch?v={video_id}"
    download_youtube_video(yt_url, video_path)


def get_workspace_paths(file_id):
    Path.mkdir(Path(os.path.join("records", file_id)), exist_ok=True)
    video_path = os.path.join("records", file_id, "video.mp4")
    audio_path = os.path.join("records", file_id, "audio.mp3")
    transcript_json_path = os.path.join("records", file_id, "transcript.json")
    transcript_txt_path = os.path.join("records", file_id, "transcript.txt")
    return video_path, audio_path, transcript_json_path, transcript_txt_path


def transcription_task(file_id: str, notion_page_id: str):
    video_path, audio_path, transcript_json_path, transcript_txt_path = get_workspace_paths(file_id)
    if os.path.exists(transcript_txt_path):
        transcript_text = read_file(transcript_txt_path)
    else:
        run_google_drive_workflow(file_id, video_path)

        extract_audio_from_video(video_path, audio_path)
        segments = transcribe(audio_path)
        transcript_text = segments_to_text(segments)
        save_data(transcript_json_path, segments)
        write_to_file(transcript_txt_path, transcript_text)
        on_transcript_finished(file_id, notion_page_id)
    return transcript_text


def on_transcript_finished(file_id, notion_page_id):
    response = requests.post("https://hook.eu2.make.com/wopgr4w7twxorkhcpqtjj3otx8qcin2v",
                             json={
                                 "transcript": get_transcribe_text_api(file_id),
                                 "notionPageId": notion_page_id
                             })
    print(response.ok)


# --------------------
# api
# --------------------


@app.get("/transcribe")
def run_transcription(file_id: str, notion_page_id: str):
    thread = Thread(target=transcription_task, args=(file_id, notion_page_id))
    thread.start()
    return {"message": "Transcription started", "file_id": file_id, "notion_page_id": notion_page_id}


@app.get("/getTranscriptText")
def get_transcribe_text_api(file_id: str):
    transcribe_text_path = f"records/{file_id}/transcript.txt"
    if os.path.exists(transcribe_text_path):
        return read_file(transcribe_text_path)
    else:
        return None


@app.get("/getTranscriptJSON")
def get_transcribe_json_api(file_id: str):
    transcribe_json_path = os.path.join("records", file_id, "transcript.json")
    if os.path.exists(transcribe_json_path):
        return read_file(transcribe_json_path)
    else:
        return None
