import numpy as np
from cv2 import cv2


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
    from algorithms.image import open_image
    # from algorithms.noise import salt_and_pepper
    # from algorithms.operator import prewitt, roberts, sobel, sharra
    # from algorithms.median import median
    # from config import BASE_DIR
    #
    # # image = open_image()
    # image = cv2.imread(f'{BASE_DIR}/test_imgs/3.png')
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # image_w, image_h = gray_image.shape
    #
    # cv2.imwrite(f'{BASE_DIR}/tests/gray_image.png', gray_image)
    # salt_and_pepper(gray_image, 50)
    # cv2.imwrite(f'{BASE_DIR}/tests/salt_and_pepper.png', gray_image)
    #
    # gray_image = median(gray_image, kernel_col=3, verbose=False, kernel_row=1)
    # gray_image = median(gray_image, kernel_col=3, verbose=False, kernel_row=1)
    # cv2.imwrite(f'{BASE_DIR}/tests/median.png', gray_image)
    #
    # ref_res = reference(gray_image)
    # cv2.imwrite(f'{BASE_DIR}/tests/ref_res.png', ref_res)
    #
    # sobel_res = sobel.sobel_edge_detection(gray_image, operator='sobel')[0]
    # cv2.imwrite(f'{BASE_DIR}/tests/sobel_res.png', sobel_res)
    #
    # prewitt_res = prewitt.prewitt_edge_detection(gray_image, operator='prewitt')[0]
    # cv2.imwrite(f'{BASE_DIR}/tests/prewitt_res.png', prewitt_res)
    #
    # roberts_res = roberts.roberts_edge_detection(gray_image, operator='roberts')[0]
    # cv2.imwrite(f'{BASE_DIR}/tests/roberts_res.png', roberts_res)
    #
    # sharra_res = sharra.sharra_edge_detection(gray_image, operator='sharra')[0]
    # cv2.imwrite(f'{BASE_DIR}/tests/sharra_res.png', sharra_res)
    #
    # check = check_image(ref_res, sobel_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # perc_sobel = round(accuracy * 100, 3)
    # print(f'Sobel = {perc_sobel}')
    #
    # check = check_image(ref_res, prewitt_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # perc_prewitt = round(accuracy * 100, 3)
    # print(f'Prewitt = {perc_prewitt}')
    #
    # check = check_image(ref_res, roberts_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # perc_roberts = round(accuracy * 100, 3)
    # print(f'Roberts = {perc_roberts}')
    #
    # check = check_image(ref_res, sharra_res)
    # count_not_zero = np.count_nonzero(check > 0)
    # accuracy = count_not_zero / (image_w * image_h)
    # perc_sharra = round(accuracy * 100, 3)
    # print(f'Sharra = {perc_sharra}')
    # -----------------------------------------
    from algorithms.image import open_image
    from algorithms.noise import salt_and_pepper
    from algorithms.operator import prewitt, roberts, sobel, sharra
    from filters.median import median

    from config import BASE_DIR

    # image = open_image()
    image = f'{BASE_DIR}/test_imgs/3.png'
    image = cv2.imread(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig_img = np.copy(gray_image)
    image_w, image_h = gray_image.shape

    gray_image = salt_and_pepper(gray_image, percent=40)

    cv2.imwrite(f'{BASE_DIR}/tests/result/salt_and_pepper.png', gray_image)

    gray_image = median(image=gray_image, kernel_col=3, kernel_row=3, verbose=False,  col_and_row=True)
    gray_image = median(image=gray_image, kernel_col=3, kernel_row=3, verbose=False, col_and_row=True)

    cv2.imwrite(f'{BASE_DIR}/tests/result/median.png', gray_image)

    print('Расчёт эталона')
    reference_res = reference(orig_img)

    print('Расчёт оператора Собеля')
    sobel_res = sobel.sobel_edge_detection(gray_image, operator='sobel')[0]
    print('Расчёт оператора Превитта')
    prewitt_res = prewitt.prewitt_edge_detection(gray_image, operator='prewitt')[0]

    print('Расчёт оператора Робертса')
    roberts_res = roberts.roberts_edge_detection(gray_image, operator='roberts')[0]

    print('Расчёт оператора Щарра\n')
    sharra_res = sharra.sharra_edge_detection(gray_image, operator='sharra')[0]

    cv2.imwrite(f'{BASE_DIR}/tests/result/reference.png', reference_res)

    cv2.imwrite(f'{BASE_DIR}/tests/result/test_sobel.png', sobel_res)
    check = check_image(reference_res, sobel_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    print(f'Проверка оператора Собеля\nпикселей не совпало: {count_not_zero}')
    print(f'Оператор Собеля не совпал на {round(accuracy * 100, 3)}%\n')
    cv2.imwrite(f'{BASE_DIR}/tests/result/test_accuracy_sobel.png', check)

    cv2.imwrite(f'{BASE_DIR}/tests/result/test_prewitt.png', prewitt_res)
    check = check_image(reference_res, prewitt_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    print(f'Проверка оператора Превитта\nпикселей не совпало: {count_not_zero}')
    print(f'Оператор Превитта не совпал на {round(accuracy * 100, 3)}%\n')
    cv2.imwrite(f'{BASE_DIR}/tests/result/test_accuracy_prewitt.png', check)

    cv2.imwrite(f'{BASE_DIR}/tests/result/test_roberts.png', roberts_res)
    check = check_image(reference_res, roberts_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    print(f'Проверка оператора Робертса\nпикселей не совпало: {count_not_zero}')
    print(f'Оператор Робертса не совпал на {round(accuracy * 100, 3)}%\n')
    cv2.imwrite(f'{BASE_DIR}/tests/result/test_accuracy_roberts.png', check)

    cv2.imwrite(f'{BASE_DIR}/tests/result/test_sharra.png', sharra_res)
    check = check_image(reference_res, sharra_res)
    count_not_zero = np.count_nonzero(check > 0)
    accuracy = count_not_zero / (image_w * image_h)
    print(f'Проверка оператора Щарра\nпикселей не совпало: {count_not_zero}')
    print(f'Оператор Щарра не совпал на {round(accuracy * 100, 3)}%\n')
    cv2.imwrite(f'{BASE_DIR}/tests/result/test_accuracy_sharra.png', check)
