
import cv2
import numpy as np
from skimage import io, util
from skimage.filters import threshold_otsu
import argparse
import os

def create_mask(image_path, side):
    """
    Loads an image, applies adaptive equalization, crops to the specified side,
    and creates a binary mask of the vertical feature.
    Returns the mask as a NumPy array.
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Adjust contrast and brightness
    adjusted = cv2.convertScaleAbs(img, alpha=1.1, beta=-10)

    # Apply adaptive equalization
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_eq = clahe.apply(adjusted)

    # Crop the specified side of the image
    height, width = img.shape
    if side == 'left':
        cropped_side = img_eq[:, :width//2]
    elif side == 'right':
        cropped_side = img_eq[:, width//2:]
    else:
        print("Error: Invalid side specified. Choose 'left' or 'right'.")
        return None

    # Threshold the image to create a mask
    thresh = threshold_otsu(cropped_side)
    mask = cropped_side > thresh
    return util.img_as_ubyte(mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a mask for a vertical feature in an image.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    parser.add_argument('side', type=str, choices=['left', 'right'], help="The side of the image to process ('left' or 'right').")
    args = parser.parse_args()

    mask = create_mask(args.image_path, args.side)
    if mask is not None:
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_filename = f"{base_name}_mask_{args.side}.png"
        io.imsave(output_filename, mask)
        print(f"Mask saved to {output_filename}")
