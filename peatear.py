#!/usr/bin/env python3

import click
import cv2
import numpy as np
import os
import pandas as pd
import sys


MINIMUM_SIZE_THRESHOLD = 350
LENGTH_RATIO_THRESHOLD = (0.4, 1.6)


@click.command()
@click.argument("image_path")
@click.argument("out_dir_path")
@click.argument("rows", type=int)
@click.argument("columns", type=int)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose mode.")
@click.version_option("1.0.0", message="%(version)s")
def cli(image_path: str, out_dir_path: str, rows: int, columns: int, verbose: bool):
    """Quick and easy data generation for ranked set sampling.
    
    Splits the image given by IMAGE_PATH into ROWS rows and COLUMNS columns.
    Outputs everything to the directory given by OUT_DIR_PATH.

    """

    if not os.path.isfile(image_path):
        print(
            "\033[91mError: Given `image_path` doesn't exist.\033[00m", file=sys.stderr
        )
        sys.exit(1)

    out_dir_path = os.path.dirname(out_dir_path)
    os.makedirs(out_dir_path, exist_ok=True)

    img = cv2.imread(image_path)
    height, width, _ = img.shape

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    img_blurred = cv2.GaussianBlur(img_hsv, (15, 15), 0)

    img_threshed_1 = cv2.inRange(img_blurred, (35, 40, 0), (100, 255, 205))
    img_threshed_2 = cv2.inRange(img_blurred, (0, 0, 160), (50, 75, 255))
    img_threshed_final = cv2.subtract(img_threshed_1, img_threshed_2)

    data = {"row": [], "column": [], "green": []}

    for i in range(rows):
        for j in range(columns):
            c1 = (height // rows * i, width // columns * j)
            c2 = (height // rows * (i + 1), width // columns * (j + 1))

            cell_img = img_threshed_final[c1[0] : c2[0], c1[1] : c2[1]]

            _, candidate_contours, _ = cv2.findContours(
                cell_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [
                c
                for c in candidate_contours
                if cv2.contourArea(c) >= MINIMUM_SIZE_THRESHOLD
            ]

            shaped_contours = []
            for c in contours:
                _, (c_width, c_height), _ = cv2.minAreaRect(c)
                length_ratio = c_width / c_height

                if LENGTH_RATIO_THRESHOLD[0] < length_ratio < LENGTH_RATIO_THRESHOLD[1]:
                    shaped_contours.append(c)

            data["row"].append(i + 1)
            data["column"].append(j + 1)
            data["green"].append(sum([cv2.contourArea(c) for c in shaped_contours]))

            test_img = img[c1[0] : c2[0], c1[1] : c2[1]]
            cv2.imwrite(
                os.path.join(out_dir_path, "{},{}.jpg".format(i + 1, j + 1)),
                cv2.drawContours(
                    test_img.copy(), shaped_contours, -1, (255, 255, 0), 2
                ),
            )

    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(out_dir_path, "ranking.csv"), float_format="%.2f", index=False
    )


if __name__ == "__main__":
    cli()
