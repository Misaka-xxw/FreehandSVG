from core.basic import ImageNp
from core.filter.basic_filter import BasicFilter
import numpy as np

class BrightContrastFilter(BasicFilter):
    def __init__(self, image: ImageNp):
        super().__init__(image)
        self.bright_factor = 0.0  # [-255,255]?
        self.contrast_factor = 0.0  # [-1,1]?

    def set_factor(self, b: float = 0.0, c: float = 0.0):
        self.bright_factor = b
        self.contrast_factor = c

    def apply(self) -> ImageNp:
        self.image.color = (self.contrast_factor + 1) * self.image.color.astype(np.float32) + self.bright_factor
        np.clip(self.image.color, 0, 255, out=self.image.color)
        return self.image