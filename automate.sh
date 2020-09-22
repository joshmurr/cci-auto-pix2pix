#!/bin/bash

set -e

download_script="./downloads.sh"

if [ -f $downloads_script ]; then
  echo "Running downloads.sh..."
  echo

  $downloads_script
fi

if [ $? -eq 0 ]; then
  clear

  echo "Finished downloading relevant programs successfully!"
  
else
  clear
  echo "There seemed to be an error in the error script!"
  exit 1
fi

#  Don't forget to export variables needed in child shell processes.

echo
echo "Enter the YouTube video ID you would like to use:"
echo

read -n 11 video_id

echo "Getting download options for video: $video_id"
echo

youtube-dl -F $video_id > tmp_youtube_info.txt

tail -n +1 tmp_youtube_info.txt

echo
echo "Enter format code for chosen video download: "

read $format_code

echo
echo "Enter output filename: "

read $filename

extension=$(tail -n +4 tmp_youtube_info.txt | awk '{print $1":" $2}' | grep $format_code | cut -d':' -f 2)
fullname="$filename.$extension"

clear
echo "Now downloading $fullname..."
echo

youtube-dl -f $format_code -o $fullname $video_id

echo
echo "Enter number of frames you would like to extract: "
read num_framees

echo "Enter output dimension (just one number, it will be square): "
read dimension

clear
echo "Now extracting frames..."

if [ -f ./cci-auto-pix2pix/frame_extractor.sh ]; then
  cci-auto-pix2pix/frame_extractor.sh $fullname ./cci-auto-pix2pix/dataset $num_frames -1:$dimension
else
  echo -e "Error finding 'frame_extractor.sh' script!\n\n"
  exit 1
fi

clear

echo "Enter number of epochs: "
read epochs

echo -e "\nEnter batch size: "
read batch_size

echo -e "\nNow running the Python model script! Good luck!...\n\n"

python3 ./cci-auto-pix2pix/main.py -n $filename -e $epochs -d ./cci-auto-pix2pix/dataset -is $dimension -os $dimension -bs $batch_size
