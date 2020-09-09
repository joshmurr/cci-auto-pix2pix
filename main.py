import argparse

parser = argparse.ArgumentParser(description="A Demo")
parser.add_argument("-n",
                    "--name",
                    help="""Name of the model. All files will be saved
                    in a directory of this name.""")
parser.add_argument("-d", "--dataset-path", help="Path to the dataset file.")
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
parser.add_argument("-buf",
                    "--buffer-size",
                    help="""Affects how uniform the shuffling is:
                    if buffer_size is greater than the number of elements
                    in the dataset, you get a uniform shuffle;
                    if it is 1 then you get no shuffling at all.
                    If the dataset is not too large (approx 1000 images) then
                    the total size of the dataset is recommended.""",
                    type=int,
                    default=1000)
# https://dmitry.ai/t/topic/100
parser.add_argument("-bs",
                    "--batch-size",
                    help="""Batch size to feed into the network when training.
                    Dependant on the amount of memory is at your disposal and
                    the size of the images being used. Defaults to 4.""",
                    type=int,
                    default=4)

if __name__ == '__main__':
    import os
    DATASET_PATH = ""
    NAME = "sample"

    args = parser.parse_args()

    if not args.name:
        print("Using name 'sample' as no name was provided.")
        print()
    else:
        NAME = args.name.replace(' ', '_')
        print(f"Using {NAME} for the model name and directories.")
        print()

    dataset_count = 0
    if args.dataset_path and os.path.exists(args.dataset_path)\
            and os.path.isdir(args.dataset_path):
        DATASET_PATH = args.dataset_path
        print("Checking files in dataset directory...")
        print()
        extensions = set()
        for file in os.listdir(DATASET_PATH):
            dataset_count += 1
            ext = os.path.splitext(file)[1]
            extensions.add(ext)
        print(f"Found {dataset_count} files with the extensions: {extensions}")
        print()

        if not os.path.exists(DATASET_PATH + '/train'):
            os.mkdir(DATASET_PATH + '/train')

        if not os.path.exists(DATASET_PATH + '/test'):
            os.mkdir(DATASET_PATH + '/test')

        four_fifths = (dataset_count // 5) * 4

        for i, image_name in enumerate(os.listdir(DATASET_PATH)):
            image_path = '/'.join((DATASET_PATH, image_name))
            train_path = '/'.join((DATASET_PATH, 'train/'))
            test_path = '/'.join((DATASET_PATH, 'test/'))
            if os.path.isfile(image_path):
                if i <= four_fifths:
                    os.rename(image_path, train_path + image_name)
                else:
                    os.rename(image_path, test_path + image_name)

    else:
        print("""
            Please provide a valid path to a directory containing the dataset.
            """)
        print()
        exit()

    if not args.batch_size:
        print("No Batch Size provided, using default of 4.")
        print()
    else:
        print(f"Using provided Batch Size of {args.batch_size}")
        print()

    if not args.buffer_size:
        BUFFER_SIZE = dataset_count if dataset_count <= 2000 else 1000
        print("No Buffer Size provided.")
        print(f"There are {dataset_count} images in the dataset, " +
              f"so using calculated Buffer Size of {BUFFER_SIZE}.")
        print()
    else:
        BUFFER_SIZE = args.buffer_size
        print(f"Using provided Buffer Size of {args.buffer_size}")
        print()

    print("Creating Dataset...")
    print()

    print(args)

    from dataset import Dataset

    Dataset = Dataset(args)
