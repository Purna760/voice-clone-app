#!/bin/bash
apt-get update
apt-get install -y libsndfile1 ffmpeg
pip install --upgrade pip
pip install -r requirements.txt
