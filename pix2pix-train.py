import argparse

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

if __name__ == '__main__':
    import os
    DATASET_PATH = ""
    NAME = "sample"
    BUFFER_SIZE = 1000
    BATCH_SIZE = 4
    IMG_SIZE = 256

    args = parser.parse_args()

    if not args.name:
        print("Using name 'sample' as no name was provided.")
        print()
    else:
        NAME = args.name.replace(' ', '_')
        print(f"Using {NAME} for the model name and directories.")
        print()

    dataset_count = 0
    if args.dataset and os.path.exists(args.dataset)\
            and os.path.isdir(args.dataset):
        DATASET_PATH = args.dataset
        print("Checking files in dataset directory...")
        print()
        extensions = set()
        for file in os.listdir(DATASET_PATH):
            dataset_count += 1
            ext = os.path.splitext(file)[1]
            extensions.add(ext)
        print(f"Found {dataset_count} files with the extensions: {extensions}")
        print()

    else:
        print("Please provide a valid path to a directory containing the\
              dataset.")
        print()

    if not args.batch_size:
        print("No Batch Size provided, using default of 4.")
        print()
    else:
        BATCH_SIZE = args.batch_size
        print(f"Using provided Batch Size of {BATCH_SIZE}")
        print()

    if not args.buffer_size:
        BUFFER_SIZE = dataset_count if dataset_count <= 2000 else 1000
        print("No Buffer Size provided.")
        print(f"There are {dataset_count} images in the dataset, " +
              f"so using calculated Buffer Size of {BUFFER_SIZE}.")
        print()
    else:
        BUFFER_SIZE = args.buffer_size
        print(f"Using provided Buffer Size of {BUFFER_SIZE}")
