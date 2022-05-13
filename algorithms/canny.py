import numpy as np
from cv2 import cv2

from algorithms.operator import sobel, prewitt, roberts
from gaussian_smoothing import gaussian_blur
from config import BASE_DIR


class Canny:
    _operators = ('sobel', 'roberts', 'prewitt')
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

    def _hysteresis(self, image, weak, operator, verbose: bool):
        image_row, image_col = image.shape

        top_to_bottom = image.copy()

        for row in range(1, image_row):
            for col in range(1, image_col):
                if top_to_bottom[row, col] == weak:
                    if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[
                        row - 1, col] == 255 or top_to_bottom[
                        row + 1, col] == 255 or top_to_bottom[
                        row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[
                        row - 1, col + 1] == 255 or top_to_bottom[
                        row + 1, col + 1] == 255:
                        top_to_bottom[row, col] = 255
                    else:
                        top_to_bottom[row, col] = 0

        bottom_to_top = image.copy()

        for row in range(image_row - 1, 0, -1):
            for col in range(image_col - 1, 0, -1):
                if bottom_to_top[row, col] == weak:
                    if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[
                        row - 1, col] == 255 or bottom_to_top[
                        row + 1, col] == 255 or bottom_to_top[
                        row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[
                        row - 1, col + 1] == 255 or bottom_to_top[
                        row + 1, col + 1] == 255:
                        bottom_to_top[row, col] = 255
                    else:
                        bottom_to_top[row, col] = 0

        right_to_left = image.copy()

        for row in range(1, image_row):
            for col in range(image_col - 1, 0, -1):
                if right_to_left[row, col] == weak:
                    if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[
                        row - 1, col] == 255 or right_to_left[
                        row + 1, col] == 255 or right_to_left[
                        row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[
                        row - 1, col + 1] == 255 or right_to_left[
                        row + 1, col + 1] == 255:
                        right_to_left[row, col] = 255
                    else:
                        right_to_left[row, col] = 0

        left_to_right = image.copy()

        for row in range(image_row - 1, 0, -1):
            for col in range(1, image_col):
                if left_to_right[row, col] == weak:
                    if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[
                        row - 1, col] == 255 or left_to_right[
                        row + 1, col] == 255 or left_to_right[
                        row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[
                        row - 1, col + 1] == 255 or left_to_right[
                        row + 1, col + 1] == 255:
                        left_to_right[row, col] = 255
                    else:
                        left_to_right[row, col] = 0

        if verbose:
            top_to_bottom[top_to_bottom > 255] = 255
            bottom_to_top[bottom_to_top > 255] = 255
            right_to_left[right_to_left > 255] = 255
            left_to_right[left_to_right > 255] = 255
            cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/hysteresis_top_to_bottom.jpg', top_to_bottom)
            cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/hysteresis_bottom_to_top.jpg', bottom_to_top)
            cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/hysteresis_right_to_left.jpg', right_to_left)
            cv2.imwrite(f'{BASE_DIR}/images/verbose/{operator}/hysteresis_left_to_right.jpg', left_to_right)

        final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

        final_image[final_image > 255] = 255

        return final_image

    def run(self, image, name_file: str, extension: str,
            is_hysteresis: bool = False,
            operator: str = 'sobel',
            kernel_size: int = 5, sigma: float = 1,
            verbose: bool = False):
        image = cv2.imread(image)
        # TODO: sigma https://youtu.be/X5O6wVmOYvk?t=1716
        blurred_image = gaussian_blur(image=image, kernel_size=kernel_size, sigma=sigma,
                                      operator=operator, verbose=verbose)
        if operator is self._operators[0]:
            gradient_magnitude, gradient_direction = sobel.sobel_edge_detection(
                blurred_image, operator, verbose=verbose
            )
        elif operator is self._operators[-1]:
            gradient_magnitude, gradient_direction = prewitt.prewitt_edge_detection(
                blurred_image, operator, verbose=verbose
            )
        elif operator is self._operators[1]:
            gradient_magnitude, gradient_direction = roberts.roberts_edge_detection(
                blurred_image, operator, verbose=verbose
            )
        else:
            raise 'Unknown operator'
        image = self._non_max_suppression(gradient_magnitude, gradient_direction, operator, verbose=verbose)
        # TODO: как выбрать Low и High?
        weak = 50
        result = self._threshold(image=image, low=5, high=20, weak=weak, operator=operator, verbose=verbose)
        if is_hysteresis:
            result = self._hysteresis(result, weak=weak, operator=operator, verbose=verbose)
        cv2.imwrite(f'{BASE_DIR}/images/result/{operator}/{name_file}_is_hist_{is_hysteresis}.{extension}', result)
