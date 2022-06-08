import math
import random

import numpy as np
from cv2 import cv2

from config import BASE_DIR


def salt_and_pepper(image: np.ndarray, percent: int) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    row, col = image.shape
    number_of_pixels = math.ceil(row * col * percent / 100)
    white_pixels = number_of_pixels // 2
    black_pixels = number_of_pixels // 2
    for _ in range(white_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        image[y_coord, x_coord] = 255

    for _ in range(black_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        image[y_coord, x_coord] = 0

    return image


def create_random_image(size, percent):
    row, col = size, size
    new_image = np.array([[255 for _ in range(col)] for _ in range(row)])

    number_of_pixels = math.ceil(row * col * percent / 100)
    black_pixels = number_of_pixels

    for _ in range(black_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        new_image[x_coord, y_coord] = 0

    cv2.imwrite(f'{BASE_DIR}/filters/random_image.png', new_image)


if __name__ == '__main__':
    create_random_image(50, percent=10)
