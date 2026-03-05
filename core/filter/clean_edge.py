import numpy as np
from PIL import Image

from skimage.color import rgb2lab
from skimage.filters import gaussian, sobel, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.util import img_as_ubyte


# =========================
# Parameters
# =========================

GAUSSIAN_SIGMA = 1.0
THRESHOLD_SCALE = 0.5
MIN_OBJECT_SIZE = 20


# =========================
# Image IO
# =========================

def load_rgb_image(path):
    """Load RGB image and normalize to [0,1]"""
    image = Image.open(path).convert("RGB")
    return np.asarray(image) / 255.0


def save_binary_image(binary_image, path):
    """Save boolean image as PNG"""
    Image.fromarray(img_as_ubyte(binary_image)).save(path)


# =========================
# Preprocessing
# =========================

def rgb_to_luminance(image):
    """
    Convert RGB image to LAB luminance channel.
    LAB L is more stable for edge detection than RGB gray.
    """
    lab = rgb2lab(image)
    return lab[:, :, 0] / 100.0


# =========================
# Edge Extraction
# =========================

def generate_edge_map(image):
    """
    Generate binary edge map using:
    Gaussian → Sobel → Otsu threshold
    """

    # Convert to luminance
    gray = rgb_to_luminance(image)

    # Smooth image to reduce noise
    smoothed = gaussian(gray, sigma=GAUSSIAN_SIGMA)

    # Compute gradient magnitude
    gradient = sobel(smoothed)

    # Automatic threshold
    threshold = threshold_otsu(gradient)

    # Binary edge map
    edges = gradient < (threshold * THRESHOLD_SCALE)

    # Remove small noise
    edges = remove_small_objects(edges, MIN_OBJECT_SIZE)

    return edges


# =========================
# Pipeline Execution
# =========================

if __name__ == "__main__":
    INPUT_IMAGE_PATH = r"D:\GitHub\FreehandSVG\test\img\3.png"
    OUTPUT_EDGE_PATH = r"D:\GitHub\FreehandSVG\test\img\clean_edges3.png"
    image = load_rgb_image(INPUT_IMAGE_PATH)
    edge_map = generate_edge_map(image)
    save_binary_image(edge_map, OUTPUT_EDGE_PATH)
    print("Edge map saved to:", OUTPUT_EDGE_PATH)
