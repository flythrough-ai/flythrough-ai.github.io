#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies for OpenCV
apt-get update
apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    python3-dev

# Create and update database schema
python update_db.py

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt