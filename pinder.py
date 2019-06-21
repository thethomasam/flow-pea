#!/usr/bin/env python3

import click
import cv2
import glob
import numpy as np
import os
import pandas as pd
import sys
import time


SQUARE_SIZE = 64
VIEWPORT_SIZE = 512


def print_intro():
    """Absolutely critical to the functionality of the script."""

    sys.stdout.write("(•_•)")
    sys.stdout.flush()
    time.sleep(0.5)

    sys.stdout.write("\r( •_•)>⌐■-■")
    sys.stdout.flush()
    time.sleep(0.5)

    sys.stdout.write("\r(⌐■_■)     ")
    sys.stdout.flush()
    time.sleep(1)

    sys.stdout.write('\r(⌐■_■) "Let\'s do this."')
    time.sleep(1)

    print()


@click.command()
@click.argument("in_dir_path")
@click.argument("out_dir_path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode.")
@click.version_option("1.0.0", message="%(version)s")
def cli(in_dir_path: str, out_dir_path: str, verbose: bool):
    """Categorize photos ultra fast.
    
    Shows images from IN_DIR_PATH then sorts them into categories in
    OUT_DIR_PATH based on the user's input.

    """

    print_intro()

    if not os.path.isdir(in_dir_path):
        print(
            "\033[91mError\033[00m: Given `in_dir_path` doesn't exist.", file=sys.stderr
        )
        sys.exit(1)

    out_dir_path = os.path.dirname(out_dir_path)
    os.makedirs(out_dir_path, exist_ok=True)
    os.makedirs(os.path.join(out_dir_path, "images"), exist_ok=True)

    data = {"path": [], "label": []}

    for img_path in glob.glob(os.path.join(in_dir_path, "*.jpg")):
        img = cv2.imread(img_path)

        pad_size = (VIEWPORT_SIZE - SQUARE_SIZE) // 2
        pad_size_pair = (pad_size, pad_size)
        padded_img = np.pad(img, (pad_size_pair, pad_size_pair, (0, 0)), "constant")

        height, width, _ = img.shape

        num_rows = height // SQUARE_SIZE
        num_columns = width // SQUARE_SIZE

        x_origin = (width - (num_columns * SQUARE_SIZE)) // 2
        y_origin = (height - (num_rows * SQUARE_SIZE)) // 2

        for i in range(num_rows):
            for j in range(num_columns):
                x1, y1 = (x_origin + (j * SQUARE_SIZE), y_origin + (i * SQUARE_SIZE))
                x2, y2 = (x1 + SQUARE_SIZE, y1 + SQUARE_SIZE)

                square_img = img[y1:y2, x1:x2]
                viewport_img = padded_img[
                    y1 : y2 + pad_size * 2, x1 : x2 + pad_size * 2
                ].copy()

                square_rect_start = VIEWPORT_SIZE // 2 - (SQUARE_SIZE // 2)
                square_rect_end = VIEWPORT_SIZE // 2 + (SQUARE_SIZE // 2)
                cv2.rectangle(
                    viewport_img,
                    (square_rect_start, square_rect_start),
                    (square_rect_end, square_rect_end),
                    (255, 0, 255),
                )

                focus_rect_start = int(VIEWPORT_SIZE // 2 - (SQUARE_SIZE // 3.5))
                focus_rect_end = int(VIEWPORT_SIZE // 2 + (SQUARE_SIZE // 3.5))
                cv2.rectangle(
                    viewport_img,
                    (focus_rect_start, focus_rect_start),
                    (focus_rect_end, focus_rect_end),
                    (255, 255, 0),
                )

                cv2.imshow("Pinder", viewport_img)

                data_path = os.path.join(
                    out_dir_path,
                    "images",
                    "{}-{},{}.jpg".format(
                        os.path.splitext(os.path.basename(img_path))[0], i, j
                    ),
                )

                while True:
                    pressed_key = cv2.waitKey(0)

                    if pressed_key == 0x7A:
                        data["path"].append(data_path)
                        data["label"].append(1)
                        cv2.imwrite(data_path, square_img)
                        break
                    elif pressed_key == 0x2F:
                        data["path"].append(data_path)
                        data["label"].append(0)
                        cv2.imwrite(data_path, square_img)
                        break

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir_path, "annotations.csv"), index=False)


if __name__ == "__main__":
    cli()
