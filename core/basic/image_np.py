import numpy as np

class ImageNp:
    def __init__(self, color: np.ndarray, alpha:np.ndarray|None=None):
        self.color = color # 黑白就是二维，彩色就是三维，shape为(height, width)或(height, width, channels)
        self.alpha = alpha # 有可能没有透明度通道，若没有则为 None
        """
        有几种数组图片大小：
        黑白灰：二维数组，shape为(height, width)
        黑白灰带透明度：三维数组，shape为(height, width, 2)，其中2表示灰度和透明度两个通道
        彩色：三维数组，shape为(height, width, 3)，其中3表示RGB三个通道
        """

    def get_color_channels(self):
        if self.color.ndim == 2:
            return 1
        elif self.color.ndim == 3:
            return self.color.shape[2]
        else:
            raise ValueError("Unsupported image color shape.")

    def has_alpha_channel(self):
        return self.alpha is not None
