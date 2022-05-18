import numpy as np
from cv2 import cv2

from algorithms.operator import sobel, prewitt, roberts, sharra
from algorithms.gaussian_smoothing import gaussian_blur
from config import BASE_DIR


class Canny:
    _operators = ('sobel', 'roberts', 'prewitt', 'sharra')
    """"опис оператору згортки в алгоритмі Кенні"""
    def _non_max_suppression(self, gradient_magnitude, gradient_direction, operator, verbose: bool = False):
        """4. Пригнічення не-максимумів"""
        image_row, image_col = gradient_magnitude.shape

        output_image = np.zeros(gradient_magnitude.shape)

        PI = 180

        for row in range(1, image_row - 1):
            for col in range(1, image_col - 1):
                direction = gradient_direction[row, col]

                if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                    before_pixel = gradient_direction[row, col - 1]
                    after_pixel = gradient_direction[row, col + 1]

                elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                    before_pixel = gradient_direction[row + 1, col - 1]
                    after_pixel = gradient_direction[row - 1, col + 1]

                elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                    before_pixel = gradient_direction[row - 1, col]
                    after_pixel = gradient_direction[row + 1, col]

                else:
                    before_pixel = gradient_direction[row - 1, col - 1]
                    after_pixel = gradient_direction[row + 1, col + 1]

                if gradient_magnitude[row, col] >= before_pixel \
                        and gradient_magnitude[row, col] >= after_pixel:
                    output_image[row, col] = gradient_magnitude[row, col]

        if verbose:
            cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/NonMaxSuppression.jpg', output_image)

        return output_image

    def _threshold(self, image, low, high, weak, operator, verbose: bool = False):
        """5. Відсіювання по граничним значенням"""
        output = np.zeros(image.shape)

        strong = 255

        strong_row, strong_col = np.where(image >= high)
        weak_row, weak_col = np.where((image <= high) & (image >= low))

        output[strong_row, strong_col] = strong
        output[weak_row, weak_col] = weak

        if verbose:
            cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/threshold.jpg', output)

        return output

    def run(self, image, name_file: str, extension: str,
            operator: str = 'sobel',
            kernel_size: int = 5, sigma: float = 1,
            verbose: bool = False):
        image = cv2.imread(image)
        blurred_image = gaussian_blur(image=image, kernel_size=kernel_size, sigma=sigma,
                                      operator=operator, verbose=verbose)
        if operator is self._operators[0]:
            gradient_magnitude, gradient_direction = sobel.sobel_edge_detection(
                blurred_image, operator, verbose=verbose
            )
        elif operator is self._operators[1]:
            gradient_magnitude, gradient_direction = roberts.roberts_edge_detection(
                blurred_image, operator, verbose=verbose
            )
        elif operator is self._operators[2]:
            gradient_magnitude, gradient_direction = prewitt.prewitt_edge_detection(
                blurred_image, operator, verbose=verbose
            )
        elif operator is self._operators[3]:
            gradient_magnitude, gradient_direction = sharra.sharra_edge_detection(
                blurred_image, operator, verbose=verbose
            )
        else:
            raise 'Unknown operator'
        image = self._non_max_suppression(gradient_magnitude, gradient_direction, operator, verbose=verbose)
        weak = 0
        result = self._threshold(image=image, low=10, high=20, weak=weak, operator=operator, verbose=verbose)  # 15 20
        cv2.imwrite(f'{BASE_DIR}/images/result/{operator}/{name_file}.{extension}', result)
