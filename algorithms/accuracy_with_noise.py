import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from config import BASE_DIR
from algorithms.test_operators import reference
from algorithms.noise import salt_and_pepper
from filters.median import median
from algorithms.test_operators import check_image
from algorithms.operator import prewitt, roberts, sobel, sharra


def check(step: int, is_median: bool, kernel_col: int, kernel_row: int = 1, count: int = 1, col_and_row: bool = True) -> dict:
    result = {
        'roberts': [],
        'prewitt': [],
        'sobel': [],
        'sharra': []
    }
    # image = open_image()
    image = f'{BASE_DIR}/test_imgs/3.png'
    image = cv2.imread(image)
    image_with_out_noise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_w, image_h = image_with_out_noise.shape
    if count == 1:
        if col_and_row:
            for noise_percent in range(0, 101, step):
                print(f'Процент зашумлённых пикселей {noise_percent}%')
                check_img = salt_and_pepper(np.copy(image_with_out_noise), percent=noise_percent)
                if is_median:
                    check_img = median(check_img, kernel_col=kernel_col, kernel_row=kernel_row, verbose=False)
                _check = check_accuracy_with_noice(check_img, image_with_out_noise, image_w, image_h)
                result['roberts'].append(_check['roberts'])
                result['prewitt'].append(_check['prewitt'])
                result['sobel'].append(_check['sobel'])
                result['sharra'].append(_check['sharra'])
        else:
            for noise_percent in range(0, 101, step):
                print(f'Процент зашумлённых пикселей {noise_percent}%')
                check_img = salt_and_pepper(np.copy(image_with_out_noise), percent=noise_percent)
                if is_median:
                    check_img = median(check_img, kernel_col=kernel_col, kernel_row=kernel_row, verbose=False, col_and_row=col_and_row)
                _check = check_accuracy_with_noice(check_img, image_with_out_noise, image_w, image_h)
                result['roberts'].append(_check['roberts'])
                result['prewitt'].append(_check['prewitt'])
                result['sobel'].append(_check['sobel'])
                result['sharra'].append(_check['sharra'])
    elif count == 2:
        if col_and_row:
            for noise_percent in range(0, 101, step):
                print(f'Процент зашумлённых пикселей {noise_percent}%')
                check_img = salt_and_pepper(np.copy(image_with_out_noise), percent=noise_percent)
                if is_median:
                    check_img = median(check_img, kernel_col=kernel_col, kernel_row=kernel_row, verbose=False)
                    check_img = median(check_img, kernel_col=kernel_col, kernel_row=kernel_row, verbose=False)
                _check = check_accuracy_with_noice(check_img, image_with_out_noise, image_w, image_h)
                result['roberts'].append(_check['roberts'])
                result['prewitt'].append(_check['prewitt'])
                result['sobel'].append(_check['sobel'])
                result['sharra'].append(_check['sharra'])
        else:
            for noise_percent in range(0, 101, step):
                print(f'Процент зашумлённых пикселей {noise_percent}%')
                check_img = salt_and_pepper(np.copy(image_with_out_noise), percent=noise_percent)
                if is_median:
                    check_img = median(check_img, kernel_col=kernel_col, kernel_row=kernel_row, verbose=False, col_and_row=col_and_row)
                _check = check_accuracy_with_noice(check_img, image_with_out_noise, image_w, image_h)
                result['roberts'].append(_check['roberts'])
                result['prewitt'].append(_check['prewitt'])
                result['sobel'].append(_check['sobel'])
                result['sharra'].append(_check['sharra'])
    else:
        raise ValueError(f'{count} > 2')
    return result


def check_accuracy_with_noice(gray_image, image_with_out_noise, image_w, image_h) -> dict:
    result = {
        'roberts': 0,
        'prewitt': 0,
        'sobel': 0,
        'sharra': 0
    }

    gray_image = np.uint8(gray_image)
    # ref_res = cv2.Canny(gray_image, 0, 255)
    ref_res = reference(image_with_out_noise)

    sobel_res = sobel.sobel_edge_detection(gray_image, operator='sobel')[0]

    prewitt_res = prewitt.prewitt_edge_detection(gray_image, operator='prewitt')[0]

    roberts_res = roberts.roberts_edge_detection(gray_image, operator='roberts')[0]

    sharra_res = sharra.sharra_edge_detection(gray_image, operator='sharra')[0]

    check = check_image(ref_res, sobel_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result['sobel'] = round(accuracy * 100, 3)

    check = check_image(ref_res, prewitt_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result['prewitt'] = round(accuracy * 100, 3)

    check = check_image(ref_res, roberts_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result['roberts'] = round(accuracy * 100, 3)

    check = check_image(ref_res, sharra_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result['sharra'] = round(accuracy * 100, 3)

    return result


def check_filter(step: int) -> dict:
    result = {
        'median_3_1_row': [],
        'median_3_1_col_and_row': [],
        'median_3_3': [],
    }
    image = f'{BASE_DIR}/test_imgs/3.png'
    image = cv2.imread(image)
    image_with_out_noise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_w, image_h = image_with_out_noise.shape

    for noise_percent in range(0, 101, step):
        print(f'Процент зашумлённых пикселей {noise_percent}%')
        check_image = salt_and_pepper(np.copy(image_with_out_noise), percent=noise_percent)
        median_3_1_row = check_filter_accuracy_with_noice(
            gray_image=median(check_image, kernel_col=3, kernel_row=1, verbose=False, col_and_row=False),
            image_with_out_noise=image_with_out_noise, image_w=image_w, image_h=image_h
        )
        median_3_1_col_and_row = check_filter_accuracy_with_noice(
            gray_image=median(check_image, kernel_col=3, kernel_row=1, verbose=False, col_and_row=True),
            image_with_out_noise=image_with_out_noise, image_w=image_w, image_h=image_h
        )
        median_3_3 = check_filter_accuracy_with_noice(
            gray_image=median(check_image, kernel_col=3, kernel_row=3, verbose=False),
            image_with_out_noise=image_with_out_noise, image_w=image_w, image_h=image_h
        )
        result['median_3_1_row'].append(median_3_1_row)
        result['median_3_1_col_and_row'].append(median_3_1_col_and_row)
        result['median_3_3'].append(median_3_3)
    return result


def check_filter_accuracy_with_noice(gray_image, image_with_out_noise, image_w, image_h) -> float:
    gray_image = np.uint8(gray_image)
    # ref_res = cv2.Canny(gray_image, 0, 255)
    ref_res = reference(image_with_out_noise)

    # roberts_res = roberts.roberts_edge_detection(gray_image, operator='roberts')[0]

    # prewitt_res = prewitt.prewitt_edge_detection(gray_image, operator='prewitt')[0]

    sobel_res = sobel.sobel_edge_detection(gray_image, operator='sobel')[0]

    # sharra_res = sharra.sharra_edge_detection(gray_image, operator='sharra')[0]

    # check = check_image(ref_res, roberts_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # result = round(accuracy * 100, 3)

    # check = check_image(ref_res, prewitt_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # result = round(accuracy * 100, 3)

    check = check_image(ref_res, sobel_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    result = round(accuracy * 100, 3)

    # check = check_image(ref_res, sharra_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # result = round(accuracy * 100, 3)

    return result


if __name__ == '__main__':
    step = 5
    # result = check(step=step, is_median=True, kernel_col=3, kernel_row=3, count=2, col_and_row=True)
    result = check_filter(step=step)
    y = list(range(0, 101, step))

    # x_roberts = result['roberts']
    # x_prewitt = result['prewitt']
    # x_sobel = result['sobel']
    # x_sharra = result['sharra']

    x_median_3_1_row = result['median_3_1_row']
    x_median_3_1_col_and_row = result['median_3_1_col_and_row']
    x_median_3_3 = result['median_3_3']

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.rcParams.update({'font.size': 14})

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 60)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    # plt.plot(y, x_roberts, label='Оператор Робертса')
    # plt.plot(y, x_prewitt, label='Оператор Превітту')
    # plt.plot(y, x_sobel, label='Оператор Собеля', linestyle='--')
    # plt.plot(y, x_sharra, label='Оператор Щарра')

    plt.plot(y, x_median_3_1_row, label='Медіани з трійок')
    plt.plot(y, x_median_3_1_col_and_row, label='Медіани з трійок дві ітерації')
    plt.plot(y, x_median_3_3, label='Медіанний', linestyle='--')

    plt.xlabel('Відсоток шуму', fontsize=14)
    plt.ylabel('δ', fontsize=20, rotation=-45)

    ax.yaxis.set_label_coords(-0.075, 0.49)
    plt.legend()
    ax.grid(True)
    plt.show()
