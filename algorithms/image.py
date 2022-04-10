from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile

from config import BASE_DIR


def get_jpg(name: str) -> JpegImageFile:
    return Image.open(BASE_DIR / f'images/{name}.jpg')


def get_png(name: str) -> PngImageFile:
    return Image.open(BASE_DIR / f'images/{name}.png')
