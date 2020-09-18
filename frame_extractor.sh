#!/bin/bash

# USAGE: 
# ./frame_extractor.sh {{INPUT FILE}} {{OUTPUT DIR}} {{NUM FRAMES FROM EACH VID}} {{SCALE}}

FILE=$1
OUTPUT_DIR=$2
NUM_FRAMES=$3
SCALE=$4
secs=0
i=0
extensions="webm|mov|mp4"
TMP=/tmp/tmp.jpg

suffix=${FILE##*.} # This trims everything from the front until a '.', greedily.

if [[ "$suffix" =~ ^(${extensions})$ ]]; then

  float_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$FILE")
  #duration=$(printf "%.0f" "$float_duration") # Takes the integer part of the number
  interval=$(bc -l <<< "$float_duration/$NUM_FRAMES")  # Use bc for float division

  echo
  echo "Video \"${FILE##*/}\" is $float_duration seconds long."
  echo "Extracting $NUM_FRAMES frames from $FILE every $interval seconds."
  echo

  if [[ -n $SCALE ]]; then # -n -- not empty string
    echo "Resizing images to a $SCALE scale."
  else
    echo "Keeping original scale."
    SCALE="iw:ih"
  fi

  ffmpeg -y -hide_banner -loglevel warning -accurate_seek -ss 00:00:00\
    -i "$FILE" -frames:v 1 -vf scale="$SCALE" ${TMP}

  w=0;
  h=0;

  {
    wh=$(identify -format '%w %h' "$TMP")
    w=$(echo $wh | cut -f1 -d' ')
    h=$(echo $wh | cut -f2 -d' ')
  } || {
    echo "Please install 'identify' which is part of the 'ImageMagick' package"
  }

  diff=$(( w - h ))

  # Extract Frames Routine:
  while [[ $i -le $NUM_FRAMES ]]
  do
    seek_to=$(date -d@"$secs" -u +%H:%M:%S.%3N)
    # Seek to given interval, save frame as .jpg
    #ffmpeg -y -hide_banner -loglevel warning -accurate_seek -ss "$seek_to"\
      #-i "$FILE" -frames:v 1 -vf scale="$SCALE" "$OUTPUT_DIR/$newname"

    newname=$(printf "image_%05d.jpg" "$i")
    ffmpeg -y -hide_banner -loglevel warning -accurate_seek -ss "$seek_to"\
      -i "$FILE" -frames:v 1 -vf scale="$SCALE" ${TMP}

    ffmpeg -y -hide_banner -loglevel warning -i ${TMP} -vf "crop=iw-${diff}:ih:0:0" "$OUTPUT_DIR/$newname"
    let i=i+1

    newname=$(printf "image_%05d.jpg" "$i")
    ffmpeg -y -hide_banner -loglevel warning -i ${TMP} -vf "crop=iw-${diff}:ih:${diff}:0" "$OUTPUT_DIR/$newname"
    let i=i+1

    secs=$(bc <<< "$secs + $interval")
  done

else
  echo "Wrong filetype, must be one of: $extensions"
fi

rm ${TMP}
