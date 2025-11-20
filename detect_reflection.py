
import cv2
import numpy as np
from skimage import io, util
from skimage.filters import threshold_otsu
import argparse
import os

def create_mask(image_path, side):
    """
    Loads an image, applies processing, crops to the specified side,
    and returns a binary mask.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    adjusted = cv2.convertScaleAbs(img, alpha=1.1, beta=-10)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_eq = clahe.apply(adjusted)

    height, width = img.shape
    if side == 'left':
        cropped_side = img_eq[:, :width//2]
    elif side == 'right':
        cropped_side = img_eq[:, width//2:]
    else:
        return None

    thresh = threshold_otsu(cropped_side)
    mask = cropped_side > thresh
    return util.img_as_ubyte(mask)

def dice_coefficient(mask1, mask2):
    """Calculates the Dice similarity coefficient between two binary masks."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)
    return 2. * intersection / total if total > 0 else 1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect and quantify reflections in an image.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    args = parser.parse_args()

    # Generate masks
    left_mask = create_mask(args.image_path, 'left')
    right_mask = create_mask(args.image_path, 'right')

    if left_mask is not None and right_mask is not None:
        # Save the initial masks
        io.imsave('left_mask.png', left_mask)
        io.imsave('right_mask.png', right_mask)
        print("Saved left_mask.png and right_mask.png")

        # Flip the left mask and save it
        flipped_left_mask = cv2.flip(left_mask, 1)
        io.imsave('flipped_left_mask.png', flipped_left_mask)
        print("Saved flipped_left_mask.png")

        # Calculate and print the similarity score
        dice_score = dice_coefficient(flipped_left_mask, right_mask)
        print(f"Dice Similarity Score: {dice_score:.4f}")

        # Create the visualization overlay
        print("Creating overlap visualization...")
        original_color_img = cv2.imread(args.image_path)
        height, width, _ = original_color_img.shape
        left_color_half = original_color_img[:, :width//2]

        # Flip the right mask to overlay it on the left side
        flipped_right_mask = cv2.flip(right_mask, 1)

        # Create a red highlight from the flipped mask
        highlight = np.zeros_like(left_color_half)
        highlight[flipped_right_mask == 255] = [0, 0, 255]  # Red in BGR

        # Blend the highlight onto the left color image
        overlay = cv2.addWeighted(left_color_half, 1, highlight, 0.5, 0)

        # Save the visualization
        io.imsave('overlap_visualization.png', overlay)
        print("Saved overlap_visualization.png")
