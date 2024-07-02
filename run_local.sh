#!/bin/bash

pip install -r requirements.txt
fastapi dev main.py --host 0.0.0.0 --port 3000
