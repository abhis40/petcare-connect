#!/bin/bash

# Create necessary directories
mkdir -p static/uploads
mkdir -p instance

# Install Python dependencies
pip install -r requirements.txt

# Run the application
exec python app.py
