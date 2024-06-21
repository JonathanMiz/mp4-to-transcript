pip install -r requirements.txt
nohup uvicorn app:app --reload &
ngrok http http://127.0.0.1:8000
