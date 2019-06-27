#!/usr/bin/env python3

import click
import cv2
import numpy as np
import os
import sys
import time

# img = cv.imread("./SampleImages/All.jpg", cv.IMREAD_COLOR)
# img = cv.medianBlur(img, 5)

# # Convert BGR to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)

# uh = 130
# us = 255
# uv = 255
# lh = 110
# ls = 50
# lv = 50
# lower_hsv = np.array([lh, ls, lv])
# upper_hsv = np.array([uh, us, uv])


def nothing(x):
    pass


@click.command()
@click.argument("image_path")
@click.version_option("1.0.0", message="%(version)s")
def cli(image_path: str):
    """A simple tool to perform HSV calibration on a given IMAGE_PATH."""
    if not os.path.isfile(image_path):
        print(
            "\033[91mError: Given `image_path` doesn't exist.\033[00m", file=sys.stderr
        )
        sys.exit(1)

    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    img_blurred = cv2.GaussianBlur(img_hsv, (15, 15), 0)

    lower_hsv = (0, 0, 0)
    upper_hsv = (0, 0, 0)

    window_name = "HSV Calibrator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("LowerH", window_name, 0, 255, nothing)
    cv2.createTrackbar("LowerS", window_name, 0, 255, nothing)
    cv2.createTrackbar("LowerV", window_name, 0, 255, nothing)

    cv2.createTrackbar("UpperH", window_name, 255, 255, nothing)
    cv2.createTrackbar("UpperS", window_name, 255, 255, nothing)
    cv2.createTrackbar("UpperV", window_name, 255, 255, nothing)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        lower_hsv = (
            cv2.getTrackbarPos("LowerH", window_name),
            cv2.getTrackbarPos("LowerS", window_name),
            cv2.getTrackbarPos("LowerV", window_name),
        )
        upper_hsv = (
            cv2.getTrackbarPos("UpperH", window_name),
            cv2.getTrackbarPos("UpperS", window_name),
            cv2.getTrackbarPos("UpperV", window_name),
        )

        img_thresh = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        # cv2.imshow(window_name, img_thresh)

        # _, contours, _ = cv2.findContours(
        #     img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )
        # cv2.imshow(
        #     window_name, cv2.drawContours(img.copy(), contours, -1, (255, 255, 0), 1)
        # )

        imask = img_thresh > 0
        out = np.zeros_like(img, np.uint8)
        out[imask] = img[imask]

        cv2.imshow(window_name, out)

        time.sleep(0.1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()
