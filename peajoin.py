#!/usr/bin/env python3

import click
import cv2
import glob
import numpy as np
import os
import pandas as pd
import sys


@click.command()
@click.argument("in_dir_path")
@click.argument("output_path")
@click.argument("csv_path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode.")
@click.version_option("1.0.0", message="%(version)s")
def cli(in_dir_path: str, output_path: str, csv_path: str, verbose: bool):
    """Rejoin some images split by PeaTear. Why not?
    
    Rejoins the images in the given IN_DIR_PATH into an image at OUTPUT_PATH.
    Outputs everything to the directory given by OUT_DIR_PATH.

    Data will be taken from the CSV given by CSV_PATH to focus on particular cells.

    """

    if not os.path.isdir(in_dir_path):
        print(
            "\033[91mError: Given `in_dir_path` doesn't exist.\033[00m", file=sys.stderr
        )
        sys.exit(1)
    if not os.path.isfile(csv_path):
        print("\033[91mError: Given `csv_path` doesn't exist.\033[00m", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cell_imgs = glob.glob(os.path.join(in_dir_path, "*.jpg"))
    num_rows = max(
        [int(os.path.basename(ci).split(".")[0].split(",")[0]) for ci in cell_imgs]
    )
    num_columns = max(
        [int(os.path.basename(ci).split(".")[0].split(",")[1]) for ci in cell_imgs]
    )

    keep_set = set()
    for index, row in pd.read_csv(csv_path).iterrows():
        keep_set.add((int(row["row"]), int(row["column"])))

    joined_img = None

    for i in range(1, num_rows + 1):
        joined_row = None

        for j in range(1, num_columns + 1):
            cell_img_path = os.path.join(in_dir_path, "{},{}.jpg".format(i, j))
            cell_img = cv2.imread(cell_img_path)

            if (i, j) not in keep_set:
                cell_img = cv2.GaussianBlur(cell_img, (105, 105), 0)

            if joined_row is not None:
                joined_row = np.concatenate((joined_row, cell_img), axis=1)
            else:
                joined_row = cell_img

        if joined_img is not None:
            joined_img = np.concatenate((joined_img, joined_row), axis=0)
        else:
            joined_img = joined_row

    cv2.imwrite(output_path, joined_img)


if __name__ == "__main__":
    cli()
