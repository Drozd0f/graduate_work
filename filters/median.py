from math import ceil

import numpy as np
from cv2 import cv2

from config import BASE_DIR
from algorithms.padded_image import padded


def median_blur(padded_image, image_row: int, image_col: int, kernel_row: int, kernel_col: int = 1):
    output = np.zeros((image_row, image_col))

    for row in range(image_row):
        for col in range(image_col):
            matrix = padded_image[row:row+kernel_row, col:col+kernel_col].reshape(kernel_row * kernel_col)
            matrix = np.sort(matrix)
            matrix_median = matrix[len(matrix) // 2]
            output[row, col] = matrix_median
    return output


def median(image, kernel_row: int, verbose: bool, operator, kernel_col: int = 0):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape

    pad_height = ceil((kernel_row - 1) / 2)
    pad_width = ceil((kernel_col - 1) / 2)

    if kernel_col == 0:
        padded_image = padded(
            image=image, pad_height=pad_height, pad_width=pad_width,
            top=False, left=True, right=True, bottom=False
        )
        output = median_blur(padded_image, image_row, image_col, kernel_row)
    elif kernel_col == kernel_row:
        padded_image = padded(
            image=image, pad_height=pad_height, pad_width=pad_width,
            top=True, left=True, right=True, bottom=True
        )
        output = median_blur(padded_image, image_row, image_col, kernel_row, kernel_col)
    else:
        raise ValueError('The size of the rows and columns in the filter must be the same')

    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/median_filter.jpg', image)

    return output


if __name__ == '__main__':
    from algorithms.noise import salt_and_pepper
    from algorithms.image import open_image

    image = open_image()
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('GrayVaDOOM.png', image)
    image = salt_and_pepper(image, percent=50)
    cv2.imwrite('salt_and_papper.png', image)
    image = median(image, 9, False, 'sobel', 9)
    cv2.imwrite('medianFilter.png', image)
