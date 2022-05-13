from tkinter import filedialog


def open_image() -> str:
    filename = filedialog.askopenfilename()
    return filename
