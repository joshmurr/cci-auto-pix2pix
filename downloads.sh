#!/bin/bash

set -e

echo "Downloading youtube-dl...\n\n"

sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl

echo
echo "Done.\n\n"

echo "Updating youtube-dl...\n\n"
youtube-dl -U

echo "Making cci-auto-pix2pix/dataset directory\n\n"

mkdir cci-auto-pix2pix/dataset
sudo chmod u+x ./cci-auto-pix2pix/frame_extractor.sh

echo "Installing bc (Bash Calculator)\n\n"

sudo apt install bc

echo "Done.\n\n"
