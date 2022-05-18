import math
import random

import numpy as np


def salt_and_pepper(image: np.ndarray, percent: int) -> np.ndarray:
    row, col = image.shape
    number_of_pixels = math.ceil(row * col * percent / 100)
    white_pixels = number_of_pixels // 2
    black_pixels = number_of_pixels // 2
    for _ in range(white_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        image[y_coord][x_coord] = 255

    for _ in range(black_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        image[y_coord][x_coord] = 0

    return image
