import numpy as np

from algorithms.convolution import convolution


def gaussian_kernel_helper(x, y, sigma):
    """фільтр Гауса"""
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (- ((x ** 2 + y ** 2) / (2 * sigma ** 2)))


def gaussian_kernel(size, sigma):
    """Гаусове ядро"""
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = gaussian_kernel_helper(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    S = np.sum(kernel_2D)
    for row in kernel_2D:
        for j in range(len(row)):
            row[j] = row[j] / S
    return kernel_2D


def gaussian_blur(image, kernel_size: int, sigma: float, operator, verbose: bool):
    kernel = gaussian_kernel(kernel_size, sigma=sigma)
    return convolution(image, kernel, operator, verbose=verbose)
