import os
import numpy as np


def text_to_img(text: str):
    text = text.split("\n")

    length_count = {}
    for row in text:
        if len(row) in length_count:
            length_count[len(row)] += 1
        else:
            length_count[len(row)] = 1

    # remove non max rows
    len_max = max(length_count, key=length_count.get)
    text = [row for row in text if len(row) == len_max]

    rows = len(text)
    cols = len_max

    # create a 2D array
    img = np.zeros((rows, cols), dtype=int)
    for i, row in enumerate(text):
        for j, char in enumerate(row):
            img[i, j] = 255 if char == "#" else 0

    return img
