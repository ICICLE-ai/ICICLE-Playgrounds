import cv2
import numpy as np
import pillow_heif

# Register HEIF opener for Pillow
pillow_heif.register_heif_opener()

# Open the HEIC value using Pillow
from PIL import Image
heic_image = Image.open("IMG_2011.HEIC")

# Convert Pillow value to NumPy image (BGR format for OpenCV)
cv_image = np.array(heic_image)
rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # Convert from RGB to BGR
cv2.imwrite("rooster_rgb.png", rgb_image)
bw_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
cv2.imwrite("grayscale.png", bw_image)
