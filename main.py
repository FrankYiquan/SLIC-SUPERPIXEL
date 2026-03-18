from skimage.io import imread
from utils import compare_slic
from slic_custom import slic_custom

if __name__ == "__main__":
    input_image_path = "images/input/brandeis_castle.jpg"
    image = imread(input_image_path)
    compare_slic(image, slic_custom, K=30000)