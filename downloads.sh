#!/bin/bash

set -e

printf "Downloading youtube-dl...\n\n"

sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl

printf "Done.\n\n"

printf "Updating youtube-dl...\n\n"
youtube-dl -U

printf "Making cci-auto-pix2pix/dataset directory\n\n"

mkdir cci-auto-pix2pix/dataset
sudo chmod u+x ./cci-auto-pix2pix/frame_extractor.sh

printf "Installing bc (Bash Calculator) and ImageMagick...\n\n"

sudo apt install bc imagemagick

printf "Done.\n\n"
