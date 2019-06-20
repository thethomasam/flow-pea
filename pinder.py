#!/usr/bin/env python3

import click
import cv2
import glob
import numpy as np
import os
import sys
import time


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

    positives = []
    negatives = []

    for img_path in glob.glob(os.path.join(in_dir_path, "*.jpg")):
        img = cv2.imread(img_path)

        cv2.imshow("Pinder", img)

        while True:
            pressed_key = cv2.waitKey(0)

            if pressed_key == 122:
                positives.append(
                    {
                        "path": img_path,
                        "rois": cv2.selectROIs(
                            "Pinder", img, fromCenter=False
                        ).tolist(),
                    }
                )
                break
            elif pressed_key == 47:
                negatives.append(
                    {"path": img_path, "rois": [[0, 0, img.shape[1], img.shape[0]]]}
                )
                break

    print("Positives: {}".format(positives))
    print("Negatives: {}".format(negatives))


if __name__ == "__main__":
    cli()
