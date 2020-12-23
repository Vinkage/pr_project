from pathlib import Path
import argparse
import numpy as np
import shutil
import os

parser = argparse.ArgumentParser(description="Tools for selecting training data")

parser.add_argument(
    "Path", metavar="/path/to/processed/dataset", type=str, help="path to dataset"
)

parser.add_argument(
    "Newpath", metavar="/path/to/new/subset_data", type=str, help="path to dataset"
)

parser.add_argument(
    "Subset",
    metavar="n",
    default=5000,
    type=int,
    help="the number of images to choose, default=5000",
)

parser.add_argument("--Random", action="store_true", help="select randomly")

parser.add_argument("--Seed", type=int, help="select randomly")


# Get argparse arguments
args = parser.parse_args()
print(args)

# Get the dataset path arg
dataset_path = Path(args.Path)

# Glob the images and labels into lists using pathlib
for split in dataset_path.glob("*"):
    if split == dataset_path / "test":
        for examples in split.glob("*"):
            if examples == split / "images":
                test_images = list(examples.glob("*"))
            if examples == split / "labels":
                test_labels = list(examples.glob("*"))
    if split == dataset_path / "train":
        for examples in split.glob("*"):
            if examples == split / "images":
                train_images = list(examples.glob("*"))
            if examples == split / "labels":
                train_labels = list(examples.glob("*"))

# Taking random subset
if args.Random:
    np.random.seed(args.Seed)
    # Make mask
    mask = np.full(len(train_images), False)
    mask[:args.Subset] = True
    np.random.shuffle(mask)

    # Images and labels subset
    train_images = np.array(sorted(train_images))[mask]
    train_labels = np.array(sorted(train_labels))[mask]

    print(type(train_images[:5]))
    print(train_images[:5])

    print(type(train_labels[:5]))
    print(train_labels[:5])


# Copy training files to new path destination
dst = Path(args.Newpath)
dst_train_img = dst / 'train' / 'images'
dst_train_lbl = dst / 'train' / 'labels'
dst_train_img.mkdir(parents=True, exist_ok=True)
dst_train_lbl.mkdir(parents=True, exist_ok=True)
# print(list(dst.glob('*/*')))

for example in zip(list(train_images), list(train_labels)):
    shutil.copy(example[0], dst_train_img)
    shutil.copy(example[1], dst_train_lbl)
