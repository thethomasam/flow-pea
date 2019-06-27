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


pos_count = 0
neg_count = 0


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

    global pos_count, neg_count

    print_intro()

    if not os.path.isdir(in_dir_path):
        print(
            "\033[91mError\033[00m: Given `in_dir_path` doesn't exist.", file=sys.stderr
        )
        sys.exit(1)

    out_dir_path = os.path.dirname(out_dir_path)
    os.makedirs(out_dir_path, exist_ok=True)
    os.makedirs(os.path.join(out_dir_path, "images"), exist_ok=True)

    cv2.namedWindow("Pinder", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Pinder", 2432 // 2, 1824 // 2)

    data = {"path": [], "label": []}

    for img_path in glob.glob(os.path.join(in_dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img_copy = img.copy()

        height, width, _ = img.shape

        def on_mouse(event: int, x: int, y: int, flags: int, param: any):
            global pos_count, neg_count
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                label = 0 if event == cv2.EVENT_LBUTTONDOWN else 1

                half_square_size = SQUARE_SIZE // 2

                if x < half_square_size:
                    x = half_square_size
                if x > width - half_square_size:
                    x = width - half_square_size
                if y < half_square_size:
                    y = half_square_size
                if y > height - half_square_size:
                    y = height - half_square_size

                x1, y1 = (x - half_square_size, y - half_square_size)
                x2, y2 = (x + half_square_size, y + half_square_size)

                print("{},{} {},{}".format(x1, y1, x2, y2))

                square_img = img_copy[y1:y2, x1:x2]

                path = os.path.join(
                    "images",
                    "{}-{}-{},{}-{},{}.jpg".format(
                        os.path.splitext(os.path.basename(img_path))[0],
                        label,
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                )

                if event == cv2.EVENT_LBUTTONDOWN:
                    neg_count += 1
                else:
                    pos_count += 1

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255) if event == cv2.EVENT_LBUTTONDOWN else (0, 255, 0),
                    thickness=4,
                )

                cv2.imshow("Pinder", img)

                data["path"].append(path)
                data["label"].append(label)
                cv2.imwrite(os.path.join(out_dir_path, path), square_img)

        cv2.imshow("Pinder", img)
        cv2.setMouseCallback("Pinder", on_mouse)

        while True:
            pressed_key = cv2.waitKey(0)

            if pressed_key == 0x20:
                if pos_count != neg_count:
                    print(
                        "Label mismatch! {} positives to {} negatives.".format(
                            pos_count, neg_count
                        )
                    )
                    continue
                pos_count = 0
                neg_count = 0
                break
            if pressed_key == 0x1B:
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(out_dir_path, "annotations.csv"), index=False)

                sys.exit()

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir_path, "annotations.csv"), index=False)


if __name__ == "__main__":
    cli()
