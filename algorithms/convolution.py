from math import ceil

import numpy as np
from cv2 import cv2

from config import BASE_DIR


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
        # print(f'Found 3 Channels : {image.shape}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = image_to_gray(image)
        # print(f'Converted to Gray Channel. Size : {image.shape}')
    else:
        # print(f'Image Shape : {image.shape}')
        pass

    # print(f'Kernel Shape : {kernel.shape}')

    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/image_to_gray.jpg', image)

    image_row, image_col = image.shape

    kernel_row, kernel_col = kernel.shape
    # row_counter = 3
    # if kernel_row == 3 or kernel_row == 2:
    #     row_counter = 0

    output = np.zeros(image.shape)

    pad_height = ceil((kernel_row - 1) / 2)
    pad_width = ceil((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if (kernel_row - 1) / 2 < 1 and (kernel_col - 1) / 2 < 1:
        # Draw pixels on top
        padded_image[0:1, 1:padded_image.shape[1] - 1] = image[0:1][::-1]

        # Draw pixels on left
        left_image = image[:, 0:1]
        padded_image[1:padded_image.shape[0] - 1, 0:1] = np.flip(left_image, axis=1)

        # Draw pixels on top-left corner
        corner_top_left = image[0:pad_height, 0:pad_width].T
        padded_image[0:pad_height, 0:pad_width] = corner_top_left

        # Draw pixels on bottom-left corner
        corner_bottom_left = image[image.shape[0] - 1:, 0:1].T
        padded_image[padded_image.shape[0] - 1:, 0:1] = corner_bottom_left

    else:
        # Draw pixels on top
        padded_image[0:pad_height, pad_width:padded_image.shape[1] - pad_width] = image[0:pad_height][::-1]

        # Draw pixels on left
        left_image = np.flip(image[:, 0:pad_width], axis=1)
        padded_image[pad_height:padded_image.shape[0] - pad_height, 0:pad_width] = np.flip(left_image, axis=1)

        # Draw pixels on right
        right_image = np.flip(image[:, image.shape[1]-pad_width:], axis=1)
        padded_image[pad_height:padded_image.shape[0] - pad_height, padded_image.shape[1] - pad_width:] = right_image

        # Draw pixels on the bottom
        bottom_image = image[image.shape[0] - pad_height:image.shape[0]:][::-1]
        padded_image[padded_image.shape[0] - pad_height:, pad_width:padded_image.shape[1] - pad_width] = bottom_image

        # Draw pixels on top-left corner
        corner_top_left = image[0:pad_height, 0:pad_width].T
        padded_image[0:pad_height, 0:pad_width] = corner_top_left

        # Draw pixels on top-right corner
        corner_top_right = np.flip(image[0:pad_height, image.shape[1] - pad_width:], axis=0)
        padded_image[0:pad_height, padded_image.shape[1] - pad_width:] = corner_top_right

        # Draw pixels on bottom-left corner
        corner_bottom_left = image[image.shape[0] - pad_height:, 0:pad_width].T
        padded_image[padded_image.shape[0] - pad_height:, 0:pad_width] = corner_bottom_left

        # Draw pixels on bottom-right corner
        corner_bottom_right = image[image.shape[0] - pad_height:, image.shape[1] - pad_width:].T
        padded_image[padded_image.shape[0] - pad_height:, padded_image.shape[1] - pad_width:] = corner_bottom_right

    for row in range(image_row):
        # counter = 0
        for col in range(image_col):
            # if row_counter != 3 and counter != 3:
            #     print('\nTest\n')
            #     print(f'OPERATOR: {operator}')
            #     print(f'PIX \n {padded_image[row, col]}')
            #     print(f'KERNEL\n{kernel}')
            #     print('IMAGE \n', padded_image[row:row + kernel_row, col:col + kernel_col])
            #     print('KERNEL * IMAGE \n', kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            #     print('SUM', np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col]))
            #     print('\nEnd test\n')
            #     counter += 1
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
        # if row_counter != 3:
        #     row_counter += 1

    # print(f'Output Image size : {output.shape}')
    # if verbose:
    #     cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/convolution.jpg', output)

    return output


if __name__ == '__main__':
    image = cv2.imread(f'{BASE_DIR}/images/VaDOOM.jpg')
    convolution(
        image,
        kernel=np.array([
        [-1, 0],
        [0, 1]
    ]),
        operator='sobel'
    )
