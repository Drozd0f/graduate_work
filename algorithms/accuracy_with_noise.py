import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

from algorithms.image import open_image
from algorithms.noise import salt_and_pepper
from filters.median import median
from algorithms.test_operators import check_image
from algorithms.operator import prewitt, roberts, sobel, sharra


def check(step: int, is_median: bool, kernel_row: int, kernel_col: int = 0) -> dict:
    result = {
        'roberts': [],
        'prewitt': [],
        'sobel': [],
        'sharra': []
    }
    # image = open_image()
    image = 'test_imgs/3.png'
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_w, image_h = gray_image.shape
    for noise_percent in range(0, 100, step):
        print(f'Процент зашумлённых пикселей {noise_percent}%')
        gray_image = salt_and_pepper(gray_image, percent=noise_percent)
        if is_median:
            gray_image = median(gray_image, kernel_row, False, 'test', kernel_col)
        _check = check_accuracy_with_noice(gray_image, image_w, image_h)
        result['roberts'].append(_check['roberts'])
        result['prewitt'].append(_check['prewitt'])
        result['sobel'].append(_check['sobel'])
        result['sharra'].append(_check['sharra'])

    return result


def check_accuracy_with_noice(gray_image, image_w, image_h) -> dict:
    result = {
        'roberts': 0,
        'prewitt': 0,
        'sobel': 0,
        'sharra': 0
    }

    canny_res = cv2.Canny(gray_image, 0, 255)

    sobel_res = sobel.sobel_edge_detection(gray_image, operator='sobel')[0]

    prewitt_res = prewitt.prewitt_edge_detection(gray_image, operator='prewitt')[0]

    roberts_res = roberts.roberts_edge_detection(gray_image, operator='roberts')[0]

    sharra_res = sharra.sharra_edge_detection(gray_image, operator='sharra')[0]

    check = check_image(canny_res, sobel_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result['sobel'] = round(accuracy * 100, 3)

    check = check_image(canny_res, prewitt_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result['prewitt'] = round(accuracy * 100, 3)

    check = check_image(canny_res, roberts_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result['roberts'] = round(accuracy * 100, 3)

    check = check_image(canny_res, sharra_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result['sharra'] = round(accuracy * 100, 3)

    return result


if __name__ == '__main__':
    step = 50
    result = check(step=step, is_median=True, kernel_row=3, kernel_col=0)

    y = list(range(0, 100, step))

    x_roberts = result['roberts']
    x_prewitt = result['prewitt']
    x_sobel = result['sobel']
    x_sharra = result['sharra']

    plt.plot(y, x_roberts, label='Оператора Робертса')
    plt.plot(y, x_prewitt, label='Оператора Превитта')
    plt.plot(y, x_sobel, label='Оператора Собеля')
    plt.plot(y, x_sharra, label='Оператора Щарра')

    plt.xlabel('Процент шума')
    plt.ylabel('Процент неточности')

    plt.legend()
    plt.grid()
    plt.show()
