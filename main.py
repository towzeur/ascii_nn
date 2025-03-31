import random
import math
import json

# ADDED
import os
import numpy as np
import PIL.Image as Image
import time

ASCII_1 = "#"

# -----------------------------------------------------------------------------


def char_to_array(char: str):
    """
    Convert a character to a 2D array
    """
    pbmarr = CHARLIST[char].split(" ")
    width = int(pbmarr[1])
    height = int(pbmarr[2])
    pbmarr = pbmarr[3:]

    ret_arr = []
    index = 0
    ret_arr.append([])

    for i, val in enumerate(pbmarr):
        ret_arr[index].append(int(val))
        if i % width == width - 1 and i < len(pbmarr) - 1:
            index += 1
            ret_arr.append([])

    return ret_arr


with open("charlist.json", "r") as f:
    CHARLIST = json.load(f)

CHAR_TO_ARRAY = {}
for char in CHARLIST:
    CHAR_TO_ARRAY[char] = char_to_array(char)


# -----------------------------------------------------------------------------


def array_to_source(array):
    width = len(array)
    height = len(array[0])
    outchar = ""

    for i in range(width):
        for j in range(height):
            outchar += ASCII_1 if array[i][j] == 1 else " "
        outchar += "\n"
    return outchar


def source_to_array(text: str):
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
            img[i, j] = 255 if char == ASCII_1 else 0

    return img


def generate_random_text(length):
    p = "ABCDEFGHIJKLMNOPQRSTUVWXYZ3456789"
    return "".join(random.choice(p) for _ in range(length))


class Captcha:
    @staticmethod
    def concat_chars(chars):
        char = []
        for i in range(len(chars[0])):
            concat_arr = []
            for j in range(len(chars)):
                concat_arr.extend(chars[j][i])
            char.append(concat_arr)
        return char

    @staticmethod
    def new_blank_char(width, height):
        return [[0 for _ in range(width)] for _ in range(height)]

    @staticmethod
    def add_filter_2_char_arr(char_arr, transforms):
        height = len(char_arr)
        width = len(char_arr[0])

        area_cond = transforms["area_cond"]

        for i in range(height):
            for j in range(width):
                y = i / height
                x = j / width
                if area_cond(x, y):
                    char_arr[i][j] = 1
        return char_arr

    @staticmethod
    def transform_char_arr(chararr, transforms):
        height = len(chararr)
        width = len(chararr[0])

        new_char = Captcha.new_blank_char(width, height)

        new_height_func = transforms["new_height"]
        new_width_func = transforms["new_width"]

        for i in range(height):
            for j in range(width):
                y = i / height
                x = j / width
                _y = new_height_func(x, y) * height
                _x = new_width_func(x, y) * width

                if 0 <= _y < height and 0 <= _x < width:
                    new_char[i][j] = chararr[int(_y)][int(_x)]
        return new_char

    @staticmethod
    def word_2_array(word: str):
        chars_arr = []
        for char in word:
            char_arr = CHAR_TO_ARRAY[char]

            # Padding top and bottom
            padding = Captcha.new_blank_char(len(char_arr[0]), len(char_arr) // 6)
            char_arr = padding + char_arr + padding

            chars_arr.append(char_arr)

        # Padding left and right
        padding = Captcha.new_blank_char(len(chars_arr[0][0]), len(chars_arr[0]))
        chars_arr.insert(0, padding)
        chars_arr.append(padding)

        # convert to 2d array
        arr = Captcha.concat_chars(chars_arr)
        assert len(set([len(char) for char in arr])) == 1
        return arr

    @staticmethod
    def augment(
        arr,
        r: float,
        r2: float,
        r3: float,
        r4: float,
    ):
        s = -1 if int(r * 2) else 1

        # distort
        if 1:
            arr = Captcha.transform_char_arr(
                arr,
                {
                    "new_height": lambda x, y: (1 - s * r3 / 10)
                    * (y - s * math.cos(x * math.pi * 2 * r2) / (15 - r * y))
                    + s * r4 / 10,
                    "new_width": lambda x, y: (1 + s * r4 / 10)
                    * (x + s * math.sin(y * math.pi * 2 * r) / (15 + r2 * x))
                    - s * r3 / 10,
                },
            )

        # add random line
        if 1:
            arr = Captcha.add_filter_2_char_arr(
                arr,
                {
                    "area_cond": lambda x, y: (
                        (
                            lambda _x, _y: math.cos(4 * x)
                            * (s / r2 * (_x**3) - s * (_x**2) / r3 + s / r4 * _x)
                        )(x - 0.5, y - 0.5)
                        < y
                        < (
                            lambda _x, _y: math.cos(4 * x)
                            * (s / r2 * (_x**3) - s * (_x**2) / r3 + s / r4 * _x)
                        )(x - 0.5, y - 0.5)
                        + 0.04
                    )
                },
            )

        return arr

    @staticmethod
    def random_augment(arr):
        # augmentation
        r = random.random()
        r2 = random.random()
        r3 = random.random()
        r4 = random.random()
        arr_aug = Captcha.augment(arr, r, r2, r3, r4)
        return arr_aug


def benchmark(n: int, sec=1):
    start_time = time.time()
    iterations = 0

    # create directory "samples" if not exists
    dirname = "samples-2"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    img_list = []

    # while time.time() - start_time < sec:
    while iterations < n:
        #
        text: str = generate_random_text(5)
        arr = Captcha.word_2_array(text)
        arr_aug = Captcha.random_augment(arr)

        # convert to numpy and source format (string)
        img = np.array(arr_aug, dtype=np.uint8) * 255
        #img = source_to_array(arr_aug)  # numpy (H, W) [0, 255]
        # print(img.shape)
        source: str = array_to_source(arr_aug)

        if 0:
            # save to "sample/{text}.txt"
            with open(f"samples/{text}.txt", "w") as f:
                f.write(source)

        if 0:
            # save to "sample/{text}.png"
            im = Image.fromarray(img.astype(np.uint8))
            im.save(f"{dirname}/{text}.png")

        if 1:
            img_list.append(img)

        iterations += 1

    # vertical concatenation
    if 1:
        # find the maximum width
        max_width = max([img.shape[1] for img in img_list])
        for i, img in enumerate(img_list):
            if img.shape[1] < max_width:
                padding = np.zeros((img.shape[0], max_width - img.shape[1]), dtype=int)
                img_list[i] = np.concatenate([img, padding], axis=1)

        img = np.concatenate(img_list, axis=0)
        im = Image.fromarray(img.astype(np.uint8))

        # create directory "concat" if not exists
        dirname = "concat"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        im.save(f"concat/concat.png")

    print(iterations)


# Example usage:
if __name__ == "__main__":
    captcha = Captcha()

    benchmark(10)
