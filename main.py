import re
import typing as t
from tkinter import filedialog

from cv2 import cv2

from algorithms.noise import salt_and_pepper
from filters.median import median
from algorithms.canny import Canny
from algorithms.operator.operators import Operators
from config import BASE_DIR


def pars_name_ext(path: str) -> t.List[str]:
    file_name = re.findall(r'\w+\.\w+$', path)
    return file_name[0].split('.')


def open_image():
    filename = filedialog.askopenfilename()
    return filename


def main():
    path = open_image()
    # path = '/home/dan/graduate_work/images/NURE.jpg'
    image = cv2.imread(path)
    # image = salt_and_pepper(image, percent=30)
    # cv2.imwrite(f'{BASE_DIR}/images/result/salt_and_pepper.jpg', image)
    image = median(image=image, kernel_col=3, kernel_row=3, verbose=False, col_and_row=True)
    image = median(image=image, kernel_col=3, kernel_row=1, verbose=True, col_and_row=True)
    # name, ext = pars_name_ext(path)
    # canny = Canny()
    # kernel_size, sigma = 5, 1.2
    # verbose = False
    # print('Выполнение оператора Робертса')
    # canny.run(
    #     image=image,
    #     name_file=name,
    #     extension=ext,
    #     operator=Operators.ROBERTS.value,
    #     kernel_size=kernel_size,
    #     sigma=sigma,
    #     verbose=verbose
    # )
    # print('Выполнение оператора Превитт')
    # canny.run(
    #     image=image,
    #     name_file=name,
    #     extension=ext,
    #     operator=Operators.PREWITT.value,
    #     kernel_size=kernel_size,
    #     sigma=sigma,
    #     verbose=verbose
    # )
    # print('Выполнение оператора Собеля')
    # canny.run(
    #     image=image,
    #     name_file=name,
    #     extension=ext,
    #     operator=Operators.SOBEL.value,
    #     kernel_size=kernel_size,
    #     sigma=sigma,
    #     verbose=verbose
    # )
    # print('Выполнение оператора Щарра')
    # canny.run(
    #     image=image,
    #     name_file=name,
    #     extension=ext,
    #     operator=Operators.SHARRA.value,
    #     kernel_size=kernel_size,
    #     sigma=sigma,
    #     verbose=verbose
    # )


if __name__ == '__main__':
    main()
