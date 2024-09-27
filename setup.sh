#!/bin/bash

# Update package list and install system dependencies
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt