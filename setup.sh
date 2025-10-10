#!/bin/bash
apt-get update
apt-get install -y libsndfile1 ffmpeg espeak-ng
pip install --upgrade pip
pip install -r requirements.txt
