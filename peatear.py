#!/usr/bin/env python3

import click
import cv2
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from math import ceil
from random import randint


SQUARE_SIZE = 64


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

    nn_model = tf.keras.models.load_model(os.path.join("models", "fababeans-64.h5"))

    img = cv2.imread(image_path)
    height, width, _ = img.shape

    data = {"row": [], "column": [], "samples": []}

    for i in range(rows):
        for j in range(columns):
            c1 = (height // rows * i, width // columns * j)
            c2 = (height // rows * (i + 1), width // columns * (j + 1))

            cell_img = img[c1[0] : c2[0], c1[1] : c2[1]]
            output_copy = cell_img.copy()

            cell_height = cell_img.shape[0]
            cell_width = cell_img.shape[1]

            num_sub_rows = cell_height // SQUARE_SIZE * 2 - 1
            num_sub_columns = cell_width // SQUARE_SIZE * 2 - 1

            x_origin = (cell_width - (ceil(num_sub_columns / 2) * SQUARE_SIZE)) // 2
            y_origin = (cell_height - (ceil(num_sub_rows / 2) * SQUARE_SIZE)) // 2

            samples = 0

            for m in range(num_sub_rows):
                for n in range(num_sub_columns):
                    x1, y1 = (
                        x_origin + (n * (SQUARE_SIZE // 2)),
                        y_origin + (m * (SQUARE_SIZE // 2)),
                    )
                    x2, y2 = (x1 + SQUARE_SIZE, y1 + SQUARE_SIZE)

                    subcell_img = cell_img[y1:y2, x1:x2] / 255.0

                    prediction = nn_model.predict(np.asarray([subcell_img]))

                    if not prediction[0][0] > prediction[0][1]:
                        cv2.circle(
                            output_copy,
                            (x1 + SQUARE_SIZE // 2, y1 + SQUARE_SIZE // 2),
                            32,
                            (0, 255, 0),
                            1,
                        )
                        samples += 1

            data["row"].append(i + 1)
            data["column"].append(j + 1)
            data["samples"].append(samples)

            cv2.imwrite(
                os.path.join(out_dir_path, "{},{}.jpg".format(i + 1, j + 1)),
                output_copy,
            )

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir_path, "ranking.csv"), index=False)


if __name__ == "__main__":
    cli()
