from math import ceil

import numpy as np
from cv2 import cv2

from config import BASE_DIR
from algorithms.padded_image import padded


def image_to_gray(image):
    """1. Перетворення кольорового зображення у відтінки сірого"""
    image_row, image_col = image.shape[0], image.shape[1]
    result = np.zeros((image_row, image_col))
    for row in range(image_row):
        for col in range(image_col):
            red, green, blue = image[row, col]
            result[row, col] = int(0.212 * red + 0.715 * green + 0.072 * blue)
    return result


def convolution(image: np.ndarray, kernel: np.ndarray, operator, verbose: bool = False) -> np.ndarray:
    """2. Видалення шумів"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/image_to_gray.jpg', image)

    image_row, image_col = image.shape

    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = ceil((kernel_row - 1) / 2)
    pad_width = ceil((kernel_col - 1) / 2)

    if (kernel_row - 1) / 2 < 1 and (kernel_col - 1) / 2 < 1:
        padded_image = padded(
            image=image, pad_height=pad_height, pad_width=pad_width,
            top=True, left=True, right=False, bottom=False
        )
    else:
        padded_image = padded(
            image=image, pad_height=pad_height, pad_width=pad_width,
            top=True, left=True, right=True, bottom=True
        )

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/convolution.jpg', image)

    return output


if __name__ == '__main__':
    image = cv2.imread(f'{BASE_DIR}/images/VaDOOM.jpg')
    convolution(
        image,
        kernel=np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        operator='sobel'
    )
