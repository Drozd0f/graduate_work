import re
import typing as t
from tkinter import filedialog

from cv2 import cv2

from algorithms.canny import Canny
from algorithms.operator.operators import Operators


def pars_name_ext(path: str) -> t.List[str]:
    file_name = re.findall(r'\w+\.\w+$', path)
    return file_name[0].split('.')


def open_image():
    filename = filedialog.askopenfilename()
    return filename


def main():
    # path = open_image()
    path = '/home/dan/graduate_work/images/NURE.jpg'
    name, ext = pars_name_ext(path)
    canny = Canny()
    kernel_size, sigma = 5, 1.2
    verbose = False
    # print('Выполнение оператора Робертса')
    # canny.run(
    #     image=path,
    #     name_file=name,
    #     extension=ext,
    #     operator=Operators.ROBERTS.value,
    #     kernel_size=kernel_size,
    #     sigma=sigma,
    #     verbose=verbose
    # )
    print('Выполнение оператора Робертса')
    canny.run(
        image=path,
        name_file=name + '_2',
        extension=ext,
        operator=Operators.PREWITT.value,
        kernel_size=kernel_size,
        sigma=sigma,
        verbose=verbose
    )
    # print('Выполнение оператора Превитта')
    # canny.run(
    #     image=path,
    #     name_file=name,
    #     extension=ext,
    #     operator=Operators.SOBEL.value,
    #     kernel_size=kernel_size,
    #     sigma=sigma,
    #     verbose=verbose
    # )
    # print('Выполнение оператора Щарра')
    # canny.run(
    #     image=path,
    #     name_file=name,
    #     extension=ext,
    #     operator=Operators.SHARRA.value,
    #     kernel_size=kernel_size,
    #     sigma=sigma,
    #     verbose=verbose
    # )
    # print('Выполнение оператора Кенни')
    # image = cv2.imread(path)
    # cv2.imwrite(f'images/result/canny/{name}.{ext}', cv2.Canny(image, 50, 100))


if __name__ == '__main__':
    main()
