#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Generate the data
python src/data/generate_data.py
