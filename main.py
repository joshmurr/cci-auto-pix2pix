import argparse
import math

parser = argparse.ArgumentParser(
    description="A class based implementation of the " +
    "Tensorflow pix2pix model.")
parser.add_argument(
    "-n",
    "--name",
    help="Name of the model. All files will be saved in a directory " +
    "of this name.",
    type=str,
    default="pix2pix")
parser.add_argument("-d",
                    "--dataset-path",
                    help="Path to the dataset file.",
                    type=str,
                    default="./dataset")
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
parser.add_argument(
    "-buf",
    "--buffer-size",
    help="Affects how uniform the shuffling is: if buffer_size " +
    "is greater than the number of elements in the dataset, you " +
    "get a uniform shuffle; if it is 1 then you get no shuffling " +
    "at all.  If the dataset is not too large (approx 1000 images) " +
    "then the total size of the dataset is recommended.",
    type=int,
    default=1000)
# https://dmitry.ai/t/topic/100
parser.add_argument(
    "-bs",
    "--batch-size",
    help="Batch size to feed into the network when training. " +
    "Dependant on the amount of memory is at your disposal and " +
    "the size of the images being used. Defaults to 4.",
    type=int,
    default=4)
parser.add_argument(
    "-e",
    "--epochs",
    help="Number of full training cycles. 50 will yield  decent " +
    "enough results with an image size of 256x256 and a dataset " +
    "of 1000 images.",
    type=int,
    default=50)
parser.add_argument(
    "-is",
    "--input-size",
    help="Dimensions of input image. Input single number which " +
    "is a power of 2",
    type=int,
    default=256)
parser.add_argument(
    "-os",
    "--output-size",
    help="Dimensions of input image. Input single number which " +
    "is a power of 2",
    type=int,
    default=256)
parser.add_argument(
    "-nb",
    "--notebook",
    help="Set to True if using an interactive Python notebook like Google " +
    "Colab to view sample output images as the model trains",
    type=bool,
    default=False)
parser.add_argument("-dum",
                    "--dummy-run",
                    help="Don't actually run a model to check arguments.",
                    type=bool,
                    default=False)


def approx_equal(num, target):
    return abs(num - target) < 5


def check_files_in_directory(_dir):
    extensions = set()
    count = 0
    for file in os.listdir(_dir):
        file_path = _dir + '/' + file
        if os.path.isfile(file_path):
            count += 1
            ext = os.path.splitext(file)[1]
            extensions.add(ext)

    return count, extensions


def print_details(args):
    print("{:>15}\t--- {}".format("PROGRAM", "DETAILS"))
    print("{:->14}\t--- {:-<14}".format('-', '-'))
    for item in vars(args):
        print(f"{item.upper():>15}\t--- {vars(args)[item]}")
    print()


def Log2(x):
    return math.log10(x) / math.log10(2)


def isPowerOfTwo(x):
    return (math.ceil(Log2(x)) == math.floor(Log2(x)))


if __name__ == '__main__':
    import os

    args = parser.parse_args()

    NAME = args.name
    DATASET_PATH = args.dataset_path

    if not isPowerOfTwo(args.input_size) or not isPowerOfTwo(args.output_size):
        print("Please stick to an input size or output size" +
              " which is a power of 2!")
        print()
        exit()

    if not args.name:
        print("Using name 'sample' as no name was provided.")
        print()
    else:
        NAME = args.name.replace(' ', '_')
        print(f"Using {NAME} for the model name and directories.")
        print()

    dataset_count = 0
    if (args.dataset_path and os.path.exists(args.dataset_path)
            and os.path.isdir(args.dataset_path)):
        DATASET_PATH = args.dataset_path
        print("Checking files in dataset directory...")
        print()

        train_path = DATASET_PATH + '/train'
        test_path = DATASET_PATH + '/test'

        if not os.path.exists(train_path):
            print(f"\t- Creating {train_path}")
            os.mkdir(train_path)

        if not os.path.exists(test_path):
            print(f"\t- Creating {test_path}")
            os.mkdir(test_path)

        files_in_dataset_root, extensions_in_root = check_files_in_directory(
            DATASET_PATH)

        files_in_dataset_train, extensions_in_train = check_files_in_directory(
            train_path)

        files_in_dataset_test, extensions_in_test = check_files_in_directory(
            test_path)

        extensions = set()
        extensions = extensions_in_root.union(extensions_in_train)
        extensions = extensions.union(extensions_in_test)

        dataset_count = (files_in_dataset_root + files_in_dataset_train +
                         files_in_dataset_test)

        if '.jpg' not in extensions:
            print("Please provide a dataset of .jpg images.")
            print()
            exit()
        else:
            print(f"\t- Found {dataset_count} files with the extensions:" +
                  f"{extensions}")
            print("\t- Only .jpg images will be used.")

        if dataset_count < 400:
            print("\t- Consider increasing the size of the dataset for " +
                  "better results. Ideally above 400 images.")

        if (len(extensions) > 1
                and not os.path.exists(DATASET_PATH + '/unused_files')):
            print(f"\t- Creating {args.dataset_path}/unused_files")
            os.mkdir(DATASET_PATH + '/unused_files')

        four_fifths = (dataset_count // 5) * 4

        if ((files_in_dataset_train and files_in_dataset_test)
                and (approx_equal(files_in_dataset_train, four_fifths)
                     and approx_equal(files_in_dataset_test,
                                      dataset_count - four_fifths))):
            print("\t* Dataset already organised!")
            print()
        else:
            print(f"\t- Splitting dataset 4:1 into {args.dataset_path}/" +
                  f"train and {args.dataset_path}/test respectively.")
            for file in os.listdir(train_path):
                file_path = train_path + '/' + file
                new_path = DATASET_PATH + '/' + file
                os.rename(file_path, new_path)
            for file in os.listdir(test_path):
                file_path = test_path + '/' + file
                new_path = DATASET_PATH + '/' + file
                os.rename(file_path, new_path)

            unused_path = DATASET_PATH + '/unused_files'

            for i, image_name in enumerate(os.listdir(DATASET_PATH)):
                image_path = DATASET_PATH + '/' + image_name
                if os.path.isfile(image_path):
                    if os.path.splitext(image_name)[1] == '.jpg':
                        if i <= four_fifths:
                            os.rename(image_path,
                                      train_path + '/' + image_name)
                        else:
                            os.rename(image_path, test_path + '/' + image_name)

                    else:
                        print(f"\t- Moving {image_path} into " +
                              "{args.dataset_path}/unused_files.")
                        os.rename(image_path, unused_path + image_name)

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

    print_details(args)

    print("Importing Tensorflow...")

    from dataset import Dataset
    print("Creating Dataset...")
    dataset = Dataset(name=NAME,
                      dataset_path=DATASET_PATH,
                      buffer_size=args.buffer_size,
                      batch_size=args.batch_size,
                      image_size=args.input_size)

    print()
    print("Creating relevenat directories to save the model, " +
          "figures and checkpoints.")

    saves_root = f"./{NAME}"
    if not os.path.exists(saves_root):
        print(f"\t- Creating {saves_root}")
        os.mkdir(saves_root)

    saves_figures = f"{saves_root}/figures"
    if not os.path.exists(saves_figures):
        print(f"\t- Creating {saves_figures}")
        os.mkdir(saves_figures)

    saves_checkpoints = f"{saves_root}/checkpoints"
    if not os.path.exists(saves_checkpoints):
        print(f"\t- Creating {saves_checkpoints}")
        os.mkdir(saves_checkpoints)

    saves_models = f"{saves_root}/models"
    if not os.path.exists(saves_models):
        print(f"\t- Creating {saves_models}")
        os.mkdir(saves_models)

    from model import Model
    print()
    print("Creating Model...")
    model = Model(dataset, saves_root, args.input_size, args.output_size,
                  args.notebook)

    print()
    print(f"!!! Model training for {args.epochs}" +
          f" EPOCH{'S' if args.epochs > 1 else ''} !!!")
    print()

    if not args.dummy_run:
        model.fit(args.epochs)

        print()
        print("Finished training!")
        print()
        print("Saving models...")

        model.save_models()
    else:
        print("~~~ DUMMY RUN ~~~")
        print()

    print("Done!")
