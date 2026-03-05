from core.filter import *
from core.basic import *
# from core.vector import skeletonize_image

if __name__=="__main__":
    io=ImageIO()
    img_root="D:\\GitHub\\FreehandSVG\\test\\img\\"
    test_imgs=["1.jpg","2.png","3.png"]
    img_name=test_imgs[1]
    sample_image = io.load_pil_np(img_root+img_name)
    # grey_filter = GreyFilter(sample_image)
    # grey_image = grey_filter.apply()
    # io.save_np_pil(grey_image, img_root+"grey_"+img_name)

    # bright_contrast_filter = BrightContrastFilter(sample_image)
    # bright_contrast_filter.set_factor(b=50, c=0.5)
    # bright_contrast_image = bright_contrast_filter.apply()
    # io.save_np_pil(bright_contrast_image, img_root+"bright_contrast_"+img_name)

    bright_contrast_filter = BrightContrastFilter(sample_image)
    bright_contrast_filter.set_factor(b=50, c=0.5)
    img = bright_contrast_filter.apply()
    grey_filter = GreyFilter(img)
    img = grey_filter.apply()
    edge_filter = EdgeFilter()
    # io.save_np_pil(edge_filter.canny(img), img_root + "canny_" + test_imgs[1])
    # io.save_np_pil(edge_filter.sobel(img), img_root + "sobel_" + test_imgs[1])
    # io.save_np_pil(edge_filter.prewitt(img), img_root + "prewitt_" + test_imgs[1])
    ## 二值化测试
    binarization_filter=BinarizationFilter(threshold=4)
    binary_image=binarization_filter.apply(edge_filter.sobel(img), is_convert=True)
    io.save_np_pil(binary_image, img_root + "binary_" + img_name)
