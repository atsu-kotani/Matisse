#!/bin/bash

# Define the base URL
BASE_URL="https://matisse.eecs.berkeley.edu/LearnedWeights/LMS/"

# Directory to save downloaded files
SAVE_DIR="Experiment/LearnedWeights/LMS2"

# Create the directory if it does not exist
mkdir -p "$SAVE_DIR"

# Use wget to download all files, excluding index.html and related files
wget -r -np -nH --cut-dirs=3 -P "$SAVE_DIR" --no-check-certificate \
     --reject "index.html*" "$BASE_URL"

echo "Download completed. Files are saved in $SAVE_DIR"