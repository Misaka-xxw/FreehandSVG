from core.filter import *
from core.basic import *

if __name__=="__main__":
    io=ImageIO()
    img_root="D:\\GitHub\\FreehandSVG\\test\\img\\"
    test_imgs=["1.jpg","2.png"]
    img=test_imgs[1]
    sample_image = io.load_pil_np(img_root+img)
    # grey_filter = GreyFilter(sample_image)
    # grey_image = grey_filter.apply()
    # io.save_np_pil(grey_image, img_root+"grey_"+test_imgs[1])
    bright_contrast_filter = BrightContrastFilter(sample_image)
    bright_contrast_filter.set_factor(b=50, c=0.5)
    bright_contrast_image = bright_contrast_filter.apply()
    io.save_np_pil(bright_contrast_image, img_root+"bright_contrast_"+img)
