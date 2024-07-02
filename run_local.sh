#!/bin/bash

python3.11 -m venv myenv
source myenv/bin/activate

pip install -r requirements.txt
fastapi dev main.py --host 0.0.0.0 --port 3000

deactivate