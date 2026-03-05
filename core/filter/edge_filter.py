from skimage import feature
from skimage import filters
from core.basic import ImageNp
class EdgeFilter:
    def sobel(self, image: ImageNp)->ImageNp:
        edges = filters.sobel(image.color)
        return ImageNp(edges, image.alpha)

    def canny(self, image):
        edges = feature.canny(image.color, sigma=8.0)
        return ImageNp(edges, image.alpha)

    def prewitt(self, image):
        edges = filters.prewitt(image.color)
        return ImageNp(edges, image.alpha)
