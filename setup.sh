#!/bin/bash
apt-get update
apt-get install -y libsndfile1 ffmpeg
pip install -r requirements.txt
