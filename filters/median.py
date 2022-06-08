from math import ceil

import numpy as np
from cv2 import cv2

from config import BASE_DIR
from algorithms.padded_image import padded


def median_blur_row(padded_image, image_row: int, image_col: int, kernel_col: int, kernel_row: int = 1):
    output = np.zeros((image_row, image_col))
    for row in range(image_row):
        for col in range(image_col):
            matrix = padded_image[row:row + kernel_row, col:col + kernel_col].reshape(kernel_row * kernel_col)
            matrix = np.sort(matrix)
            matrix_median = matrix[len(matrix) // 2]
            output[row, col] = matrix_median
    return output


def median_blur_col(padded_image, image_row: int, image_col: int):
    output = np.zeros((image_row, image_col))
    for col in range(image_col):
        for row in range(image_row):
            matrix = padded_image[row:row + 3, col:col + 1].reshape(3)
            matrix = np.sort(matrix)
            matrix_median = matrix[len(matrix) // 2]
            output[row, col] = matrix_median
    return output


def median(image, kernel_col: int, verbose: bool, kernel_row: int = 1, col_and_row: bool = True):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape

    pad_height = ceil((kernel_row - 1) / 2)
    pad_width = ceil((kernel_col - 1) / 2)

    if kernel_row == 1:
        padded_image = padded(
            image=image, pad_height=pad_height, pad_width=pad_width,
            top=False, left=True, right=True, bottom=False
        )
        output = median_blur_row(
            padded_image=padded_image,
            image_row=image_row,
            image_col=image_col,
            kernel_col=kernel_col
        )
        if col_and_row:
            pad_height, pad_width = pad_width, pad_height
            padded_image = padded(
                image=output, pad_height=pad_height, pad_width=pad_width,
                top=True, left=False, right=False, bottom=True
            )
            output = median_blur_col(
                padded_image=padded_image,
                image_row=image_row,
                image_col=image_col
            )
    elif kernel_col == kernel_row:
        padded_image = padded(
            image=image, pad_height=pad_height, pad_width=pad_width,
            top=True, left=True, right=True, bottom=True
        )
        output = median_blur_row(
            padded_image=padded_image,
            image_row=image_row,
            image_col=image_col,
            kernel_row=kernel_row,
            kernel_col=kernel_col
        )
    else:
        raise ValueError('The size of the rows and columns in the filter must be the same')

    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/result/median_filter.png', output)

    return output


if __name__ == '__main__':
    from algorithms.noise import salt_and_pepper
    from algorithms.image import open_image

    image = open_image()
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    image_w, image_h = image.shape
    # cv2.imwrite('orig.png', image)
    # image = salt_and_pepper(image, percent=20)
    # cv2.imwrite('salt_and_papper.png', image)
    image = median(image, kernel_col=3, verbose=False, kernel_row=3)
    cv2.imwrite('filter_image_median.png', image)
