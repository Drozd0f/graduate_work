from cv2 import cv2
import numpy as np

from algorithms.convolution import convolution
from config import BASE_DIR


def get_roberts_operator():
    return np.array([
        [-1, 0],
        [0, 1]
    ])


# TODO: там знаменатели одинаковые в знаменателе, это опечатка?
def roberts_edge_detection(image, operator, convert_to_degree=False, verbose: bool = False):
    """3. Пошук градієнту зміни яскравості в. Оператор Робертса"""
    new_image_x = convolution(image, get_roberts_operator(), operator=operator)
    new_image_y = convolution(image, np.flip(get_roberts_operator(), axis=1), operator=operator)

    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/edge_detection_new_image_x.jpg', new_image_x)
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/edge_detection_new_image_y.jpg', new_image_y)

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude[gradient_magnitude > 255] = 255

    gradient_direction = np.arctan2(new_image_y, new_image_x)

    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180

    if verbose:
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/edge_detection_result_gradient_magnitude.jpg', gradient_magnitude)
        cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/edge_detection_result_gradient_direction.jpg', gradient_direction)

    return gradient_magnitude, gradient_direction
