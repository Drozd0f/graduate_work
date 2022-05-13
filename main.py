import re
import typing as t
from tkinter import filedialog

from algorithms.canny import Canny
from algorithms.operator.operators import Operators


def pars_name_ext(path: str) -> t.List[str]:
    file_name = re.findall(r'\w+\.\w+$', path)
    return file_name[0].split('.')


def open_image():
    filename = filedialog.askopenfilename()
    return filename


def main():
    path = open_image()
    name, ext = pars_name_ext(path)
    canny = Canny()
    canny.run(
        image=path,
        name_file=name,
        extension=ext,
        is_hysteresis=False,
        operator=Operators.PREWITT.value,
        kernel_size=5,
        sigma=1.2,
        verbose=True
    )


if __name__ == '__main__':
    main()
