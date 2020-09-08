#!/usr/bin/env python3

#  import numpy as np
#  import tensorflow as tf
#  from tensorflow import keras
#  import os
#  import time
#  from matplotlib import pyplot as plt
import argparse
#  import math
#  import random

DATASET_PATH = ""
NAME = "sample"
BUFFER_SIZE = 1000
BATCH_SIZE = 4
IMG_SIZE = 256

parser = argparse.ArgumentParser(description="A Demo")
parser.add_argument("-n",
                    "--name",
                    help="""Name of the model. All files will be saved
                    in a directory of this name.""")
parser.add_argument("-d", "--dataset", help="Path to the dataset file.")
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
parser.add_argument("-buf",
                    "--buffer-size",
                    help="""Affects how uniform the shuffling is:
                    if buffer_size is greater than the number of elements
                    in the dataset, you get a uniform shuffle;
                    if it is 1 then you get no shuffling at all.
                    If the dataset is not too large (approx 1000 images) then
                    the total size of the dataset is recommended.""",
                    type=int)
# https://dmitry.ai/t/topic/100
parser.add_argument("-bs",
                    "--batch-size",
                    help="""Batch size to feed into the network when training.
                    Dependant on the amount of memory is at your disposal and
                    the size of the images being used. Defaults to 4.""",
                    type=int)
parser.add_argument("-s",
                    "--image-size",
                    help="""Width and Height of the image, must be square.""")

if __name__ == '__main__':
    import os

    args = parser.parse_args()

    if not args.name:
        print("Using name 'sample' as no name was provided.")
        print()

    if args.dataset:
        if os.path.exists(args.dataset) and os.path.isdir(args.dataset):
            DATASET_PATH = args.dataset
            print("Checking files in dataset directory...")
            count = 0
            extensions = set()
            for file in os.listdir(DATASET_PATH):
                count += 1
                ext = os.path.splitext(file)[1]
                extensions.add(ext)

            print(f"Found {count} files with the extensions: {extensions}")
