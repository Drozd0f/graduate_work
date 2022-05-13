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
        print(f'Found 3 Channels : {image.shape}')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image_to_gray(image)
        print(f'Converted to Gray Channel. Size : {image.shape}')
    else:
        print(f'Image Shape : {image.shape}')

    print(f'Kernel Shape : {kernel.shape}')

    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/image_to_gray.jpg', image)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    print(f'Output Image size : {output.shape}')
    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/convolution.jpg', output)

    return output
