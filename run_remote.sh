#!/bin/bash

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
ngrok http http://127.0.0.1:8000