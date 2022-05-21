import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

from algorithms.image import open_image
from algorithms.noise import salt_and_pepper
from algorithms.operator import prewitt, roberts, sobel, sharra


# значение (reference + operator) % 510
# p = np.count_nonzero(((reference + operator) % 510) > 0) / (n * m) где n, m - размер изобр
# p * log(p)


def check_image(refer_img, operat_img):
    width, height = refer_img.shape
    new_image = np.zeros((width, height))
    for col in range(height):
        for row in range(width):
            new_image[row, col] = (refer_img[row, col] + operat_img[row, col]) % 510
    return new_image


def horizontal(image: np.ndarray) -> np.ndarray:
    width, height = image.shape
    new_image = np.zeros((width, height))
    for col in range(height):
        for row in range(width):
            if image[row, col] != 255:
                new_image[row, col] = 255
    return new_image


def vertical(image: np.ndarray) -> np.ndarray:
    width, height = image.shape
    new_image = np.zeros((width, height))
    for row in range(width):
        for col in range(height):
            if image[row, col] != 255:
                new_image[row, col] = 255
    return new_image


def reference(image: np.ndarray):
    new_image_x = horizontal(image)
    new_image_y = vertical(image)
    result = (new_image_x + new_image_y)
    result[result > 255] = 255
    return result


if __name__ == '__main__':
    image = open_image()
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(type(gray_image[400:401]))

    # image_w, image_h = gray_image.shape
    # _try = 3
    #
    # gray_image = salt_and_pepper(gray_image, percent=10)
    # cv2.imwrite(f'test_imgs/result/{_try}/orig_image_with_noise.png', gray_image)
    #
    # print('Расчёт Кенни')
    # canny_res = cv2.Canny(gray_image, 0, 255)
    #
    # print('Расчёт эталона')
    # reference_res = reference(gray_image)
    #
    # print('Расчёт оператора Собеля')
    # sobel_res = sobel.sobel_edge_detection(gray_image, operator='sobel')[0]
    #
    # print('Расчёт оператора Превитта')
    # prewitt_res = prewitt.prewitt_edge_detection(gray_image, operator='prewitt')[0]
    #
    # print('Расчёт оператора Робертса')
    # roberts_res = roberts.roberts_edge_detection(gray_image, operator='roberts')[0]
    #
    # print('Расчёт оператора Щарра\n')
    # sharra_res = sharra.sharra_edge_detection(gray_image, operator='sharra')[0]
    #
    # cv2.imwrite(f'test_imgs/result/{_try}/reference.png', reference_res)
    #
    # cv2.imwrite(f'test_imgs/result/{_try}/canny.png', canny_res)
    #
    # cv2.imwrite(f'test_imgs/result/{_try}/test_sobel.png', sobel_res)
    # check = check_image(reference_res, sobel_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # print(f'Проверка оператора Собеля\nпикселей не совпало: {count_not_zero}')
    # print(f'Оператор Собеля не совпал на {round(accuracy * 100, 3)}%\n')
    # cv2.imwrite(f'test_imgs/result/{_try}/test_accuracy_sobel.png', check)
    #
    # check = check_image(canny_res, sobel_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # print(f'Проверка оператора Собеля с Кенни\nпикселей не совпало: {count_not_zero}')
    # print(f'Оператор Собеля не совпал на {round(accuracy * 100, 3)}%\n')
    # cv2.imwrite(f'test_imgs/result/{_try}/test_accuracy_sobel_canny.png', check)
    #
    # cv2.imwrite(f'test_imgs/result/{_try}/test_prewitt.png', prewitt_res)
    # check = check_image(reference_res, prewitt_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # print(f'Проверка оператора Превитта\nпикселей не совпало: {count_not_zero}')
    # print(f'Оператор Превитта не совпал на {round(accuracy * 100, 3)}%\n')
    # cv2.imwrite(f'test_imgs/result/{_try}/test_accuracy_prewitt.png', check)
    #
    # check = check_image(canny_res, prewitt_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # print(f'Проверка оператора Превитта с Кенни\nпикселей не совпало: {count_not_zero}')
    # print(f'Оператор Собеля не совпал на {round(accuracy * 100, 3)}%\n')
    # cv2.imwrite(f'test_imgs/result/{_try}/test_accuracy_prewitt_canny.png', check)
    #
    # cv2.imwrite(f'test_imgs/result/{_try}/test_roberts.png', roberts_res)
    # check = check_image(reference_res, roberts_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # print(f'Проверка оператора Робертса\nпикселей не совпало: {count_not_zero}')
    # print(f'Оператор Робертса не совпал на {round(accuracy * 100, 3)}%\n')
    # cv2.imwrite(f'test_imgs/result/{_try}/test_accuracy_roberts.png', check)
    #
    # check = check_image(canny_res, roberts_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # print(f'Проверка оператора Робертса с Кенни\nпикселей не совпало: {count_not_zero}')
    # print(f'Оператор Собеля не совпал на {round(accuracy * 100, 3)}%\n')
    # cv2.imwrite(f'test_imgs/result/{_try}/test_accuracy_roberts_canny.png', check)
    #
    # cv2.imwrite(f'test_imgs/result/{_try}/test_sharra.png', sharra_res)
    # check = check_image(reference_res, sharra_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # print(f'Проверка оператора Щарра\nпикселей не совпало: {count_not_zero}')
    # print(f'Оператор Щарра не совпал на {round(accuracy * 100, 3)}%\n')
    # cv2.imwrite(f'test_imgs/result/{_try}/test_accuracy_sharra.png', check)
    #
    # check = check_image(canny_res, sharra_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # print(f'Проверка оператора Щарра с Кенни\nпикселей не совпало: {count_not_zero}')
    # print(f'Оператор Собеля не совпал на {round(accuracy * 100, 3)}%')
    # cv2.imwrite(f'test_imgs/result/{_try}/test_accuracy_sharra_canny.png', check)
