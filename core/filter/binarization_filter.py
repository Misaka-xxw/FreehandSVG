import numpy as np
from scipy.ndimage import gaussian_filter

from core.basic import ImageNp
class BinarizationFilter:
    def __init__(self, threshold: int=128):
        self.threshold = threshold

    def apply(self, image:ImageNp, is_convert:bool=False) -> np.ndarray:
        if len(image.color.shape) == 2:  # Grayscale image
            image.color = gaussian_filter(image.color, 1.0)
            if is_convert:
                binary_image = (image.color < self.threshold).astype(np.uint8) * 255
            else:
                binary_image = (image.color > self.threshold).astype(np.uint8) * 255
            return ImageNp(binary_image, image.alpha)
        elif len(image.color.shape) == 3 and image.color.shape[2] == 3:  # Color image
            gray_image = np.dot(image.color[...,:3], [0.299, 0.587, 0.114])
            if is_convert:
                binary_image = (gray_image.color < self.threshold).astype(np.uint8) * 255
            else:
                binary_image = (gray_image.color > self.threshold).astype(np.uint8) * 255
            return ImageNp(binary_image, image.alpha)

        raise ValueError("Unsupported image format.")