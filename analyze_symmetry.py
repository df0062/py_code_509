
import cv2
import numpy as np
from skimage import io, util
from skimage.filters import threshold_otsu
import argparse
import os

def create_mask(image_chunk):
    """
    Takes an image chunk, applies thresholding and noise cleanup,
    and returns a binary mask.
    """
    if image_chunk.size == 0:
        return np.array([])

    thresh = threshold_otsu(image_chunk)
    mask = image_chunk > thresh

    # Clean up noise using morphological opening
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(util.img_as_ubyte(mask), cv2.MORPH_OPEN, kernel)

    return cleaned_mask

def dice_coefficient(mask1, mask2):
    """Calculates the Dice similarity coefficient between two binary masks."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)
    return 2. * intersection / total if total > 0 else 1.0

def analyze_vertical_reflection(processed_image):
    """Analyzes the vertical reflection symmetry of an image."""
    height, width = processed_image.shape
    left_half = processed_image[:, :width//2]
    right_half = processed_image[:, width//2:]

    left_mask = create_mask(left_half)
    right_mask = create_mask(right_half)

    flipped_right_mask = cv2.flip(right_mask, 1)

    return dice_coefficient(left_mask, flipped_right_mask)

def analyze_horizontal_reflection(processed_image):
    """Analyzes the horizontal reflection symmetry of an image."""
    height, width = processed_image.shape
    top_half = processed_image[:height//2, :]
    bottom_half = processed_image[height//2:, :]

    top_mask = create_mask(top_half)
    bottom_mask = create_mask(bottom_half)

    flipped_bottom_mask = cv2.flip(bottom_mask, 0)

    return dice_coefficient(top_mask, flipped_bottom_mask)

def analyze_horizontal_translation(processed_image):
    """Finds the best horizontal translation score."""
    height, width = processed_image.shape
    left_half = processed_image[:, :width//2]
    right_half = processed_image[:, width//2:]

    left_mask = create_mask(left_half)
    right_mask = create_mask(right_half)

    best_score = -1
    best_shift = 0

    for shift in range(-width//2, width//2):
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        translated_right_mask = cv2.warpAffine(right_mask, M, (width//2, height))
        score = dice_coefficient(left_mask, translated_right_mask)
        if score > best_score:
            best_score = score
            best_shift = shift

    return best_score, best_shift

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze the symmetry of an image.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        exit()

    # Pre-process the image once
    adjusted = cv2.convertScaleAbs(img, alpha=1.1, beta=-10)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    processed_image = clahe.apply(adjusted)

    print(f"Symmetry Analysis for: {os.path.basename(args.image_path)}")
    print("-" * 40)

    # Vertical Reflection
    v_reflect_score = analyze_vertical_reflection(processed_image)
    print(f"Vertical Reflection Score:   {v_reflect_score:.4f}")

    # Horizontal Reflection
    h_reflect_score = analyze_horizontal_reflection(processed_image)
    print(f"Horizontal Reflection Score: {h_reflect_score:.4f}")

    # Horizontal Translation
    h_trans_score, h_trans_shift = analyze_horizontal_translation(processed_image)
    print(f"Horizontal Translation Score: {h_trans_score:.4f} (at {h_trans_shift}px shift)")
    print("-" * 40)
