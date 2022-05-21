from cv2 import cv2
import numpy as np


def padded(image, pad_height, pad_width, top: bool, left: bool, right: bool, bottom: bool):
    image_row, image_col = image.shape
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if top:
        # Draw pixels on top
        padded_image[0:pad_height, pad_width:padded_image.shape[1] - pad_width] = image[0:pad_height][::-1]
        if left:
            # Draw pixels on top-left corner
            corner_top_left = image[0:pad_height, 0:pad_width].T
            padded_image[0:pad_height, 0:pad_width] = corner_top_left
        if right:
            # Draw pixels on top-right corner
            corner_top_right = np.flip(image[0:pad_height, image.shape[1] - pad_width:], axis=0)
            padded_image[0:pad_height, padded_image.shape[1] - pad_width:] = corner_top_right

    if left:
        # Draw pixels on left
        left_image = np.flip(image[:, 0:pad_width], axis=1)
        padded_image[pad_height:padded_image.shape[0] - pad_height, 0:pad_width] = np.flip(left_image, axis=1)

    if right:
        # Draw pixels on right
        right_image = np.flip(image[:, image.shape[1] - pad_width:], axis=1)
        padded_image[pad_height:padded_image.shape[0] - pad_height, padded_image.shape[1] - pad_width:] = right_image

    if bottom:
        # Draw pixels on the bottom
        bottom_image = image[image.shape[0] - pad_height:image.shape[0]:][::-1]
        padded_image[padded_image.shape[0] - pad_height:, pad_width:padded_image.shape[1] - pad_width] = bottom_image
        if left:
            # Draw pixels on bottom-left corner
            corner_bottom_left = image[image.shape[0] - pad_height:, 0:pad_width].T
            padded_image[padded_image.shape[0] - pad_height:, 0:pad_width] = corner_bottom_left
        if right:
            # Draw pixels on bottom-right corner
            corner_bottom_right = image[image.shape[0] - pad_height:, image.shape[1] - pad_width:].T
            padded_image[padded_image.shape[0] - pad_height:, padded_image.shape[1] - pad_width:] = corner_bottom_right

    return padded_image
