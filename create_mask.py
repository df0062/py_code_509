
import cv2
import numpy as np
from skimage import exposure, io, util
from skimage.filters import threshold_otsu

# Load the image
img = cv2.imread('/tmp/file_attachments/ltg_video_int_img.png', cv2.IMREAD_GRAYSCALE)

# Apply adaptive equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_eq = clahe.apply(img)

# Crop the right hand side of the image
height, width = img.shape
right_side = img_eq[:, width//2:]

# Threshold the image to create a mask
thresh = threshold_otsu(right_side)
mask = right_side > thresh

# Save the mask
io.imsave('mask.png', util.img_as_ubyte(mask))
