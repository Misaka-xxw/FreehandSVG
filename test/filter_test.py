from core.filter import *
from core.basic import *

if __name__=="__main__":
    io=ImageIO()
    img_root="D:\\GitHub\\FreehandSVG\\test\\img\\"
    test_imgs=["1.jpg","2.png"]
    sample_image = io.load_pil_np(img_root+test_imgs[1])
    grey_filter = GreyFilter(sample_image)
    grey_image = grey_filter.apply()
    io.save_np_pil(grey_image, img_root+"grey_"+test_imgs[1])