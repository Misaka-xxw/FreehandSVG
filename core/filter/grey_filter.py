from core.basic import ImageNp
from core.filter.basic_filter import BasicFilter
import numpy as np

class GreyFilter(BasicFilter):
    def __init__(self, image: ImageNp):
        super().__init__(image)
        self.rgb_rate=[0.299,0.587,0.114] # Human eye sensitivity to RGB

    def set_rgb_rate(self, r: float, g: float, b: float):
        total = r + g + b
        if total == 0:
            raise ValueError("The sum of r, g, and b must be greater than 0.")
        self.rgb_rate = [r / total, g / total, b / total]

    def apply(self) -> ImageNp:
        channel_num=self.image.get_color_channels()
        if channel_num == 1:
            return self.image
        elif channel_num==3:
            r, g, b = self.image.color[:, :, 0], self.image.color[:, :, 1], self.image.color[:, :, 2]
            grey = (r * self.rgb_rate[0] + g * self.rgb_rate[1] + b * self.rgb_rate[2])
            return ImageNp(grey,self.image.alpha)
        else:
            raise ValueError("Unsupported image color type.")
