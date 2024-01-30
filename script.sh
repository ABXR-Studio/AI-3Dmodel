#!/bin/bash

. venv/bin/activate

uvicorn main:app --host 0.0.0.0 --port 8000 --reload > /dev/null 2>&1 &

echo "ShapE PID : $!"  > save_pid.txt
