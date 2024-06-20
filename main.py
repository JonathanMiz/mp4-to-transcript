import json
import os.path
from pathlib import Path

import gdown
from fastapi import FastAPI
from faster_whisper import WhisperModel
from pydub import AudioSegment

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


model = WhisperModel("medium", device="cpu", compute_type="int8")


def transcribe(audio_file):
    segments, info = model.transcribe(audio_file, word_timestamps=True, language="he")

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


def save_data_locally(file_path, data):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)


def segments_to_text(segments):
    text = ""
    for segment in segments:
        print(segment)
        text += segment["text"] + " "
    return text


def write_to_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(data)


def read_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        return text

# --------------------
# api
# --------------------


@app.get("/run")
def run_workflow(file_id: str):
    Path.mkdir(Path(os.path.join("records", file_id)), exist_ok=True)

    video_path = os.path.join("records", file_id, "video.mp4")
    audio_path = os.path.join("records", file_id, "audio.mp3")
    transcript_json_path = os.path.join("records", file_id, "transcript.json")
    transcript_txt_path = os.path.join("records", file_id, "transcript.txt")

    g_drive_url = f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link"
    download_drive_file(g_drive_url, video_path)
    extract_audio_from_video(video_path, audio_path)

    segments = transcribe(audio_path)

    transcript_text = segments_to_text(segments)

    save_data_locally(transcript_json_path, segments)
    write_to_file(transcript_txt_path, transcript_text)
    return transcript_text


@app.get("/getTranscriptText")
def transcribe_text_api(file_id: str):
    return read_file(f"records/{file_id}/transcript.txt")


@app.get("/getTranscriptJSON")
def transcribe_json_api(file_id: str):
    return read_file(f"records/{file_id}/transcript.json")