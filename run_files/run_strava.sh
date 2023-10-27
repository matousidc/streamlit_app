#!/bin/bash

# Get the current date and time in the desired format
current_datetime=$(date "+%Y-%m-%d %H:%M:%S")
# Print the current date and time
echo ________________
echo $current_datetime
source /home/matou/projects/streamlit_app/.venv/bin/activate
python /home/matou/projects/streamlit_app/refreshing_tokens.py
