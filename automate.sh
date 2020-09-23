#!/bin/bash

set -e

echo -e "Downloading youtube-dl...\n\n"

sudo wget https://yt-dl.org/downloads/latest/youtube-dl -O /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl

echo -e "Done.\n\n"

echo -e "Updating youtube-dl...\n\n"
youtube-dl -U

echo -e "Making cci-auto-pix2pix/dataset directory\n\n"

mkdir cci-auto-pix2pix/dataset
sudo chmod u+x ./cci-auto-pix2pix/frame_extractor.sh

echo -e "Installing bc (Bash Calculator) and ImageMagick...\n\n"

sudo apt install bc imagemagick

echo -e "Done.\n\n"

if [ $? -eq 0 ]; then
  clear

  echo "Finished downloading relevant programs successfully!"
  
else
  clear
  echo "There seemed to be an error in the error script!"
  exit 1
fi

#  Don't forget to export variables needed in child shell processes.

echo -e "\nEnter the YouTube video ID you would like to use: "

read -n 11 video_id

echo -e "\nGetting download options for video: $video_id\n"

youtube-dl -F $video_id > tmp_youtube_info.txt

tail -n +1 tmp_youtube_info.txt

echo -e "\nEnter format code for chosen video download: "

read format_code

echo -e "\nEnter output filename: "

read filename

if [ -z $filename ]; then # true if empty string
  echo -e "\nPlease re-enter filename: "
  read filename
fi

if [ -z $format_code ]; then
  echo -e "\nPlease re-enter format code: "
  read format_code
fi

extension=$(tail -n +4 tmp_youtube_info.txt | awk '{print $1":" $2}' | grep $format_code | cut -d':' -f 2)

#echo -e "Filename: $filename\n"
#echo -e "Extension: $extension\n"
fullname="${filename}.${extension}"

#clear
echo -e "Now downloading ${fullname}...\n"

youtube-dl -f $format_code -o $fullname $video_id

echo -e "\nEnter number of frames you would like to extract: "
read num_framees

echo "Enter output dimension (just one number, it will be square): "
read dimension

clear
echo "Now extracting frames..."

if [ -f ./frame_extractor.sh ]; then
  ./frame_extractor.sh ${fullname} ./dataset ${num_frames} -1:${dimension}
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

python3 ./main.py -n ${filename} -e ${epochs} -d ./dataset -is ${dimension} -os ${dimension} -bs ${batch_size}
