
import cv2
import numpy as np
from skimage import io, util
from skimage.filters import threshold_otsu
import argparse
import os

def create_mask(image_path, side):
    """
    Loads an image, applies processing, crops to the specified side,
    and returns a binary mask. For the right side, it focuses on the
    right-most 25% of the image to isolate the feature.
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
        thresh = threshold_otsu(cropped_side)
        mask = cropped_side > thresh

        # Clean up noise using morphological opening
        kernel = np.ones((3,3), np.uint8)
        cleaned_mask = cv2.morphologyEx(util.img_as_ubyte(mask), cv2.MORPH_OPEN, kernel)

        return cleaned_mask
    elif side == 'right':
        # Create a full-sized black canvas for the right half
        right_mask_canvas = np.zeros((height, width//2), dtype=np.uint8)

        # Focus on the right-most 25% of the image to generate the mask
        feature_area = img_eq[:, width*3//4:]
        thresh = threshold_otsu(feature_area)
        small_mask = feature_area > thresh

        # Place the small mask onto the canvas at its correct positional offset
        start_col = width // 4
        right_mask_canvas[:, start_col:] = util.img_as_ubyte(small_mask)

        return right_mask_canvas
    else:
        return None

def dice_coefficient(mask1, mask2):
    """Calculates the Dice similarity coefficient between two binary masks."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)
    return 2. * intersection / total if total > 0 else 1.0

def find_best_translation(left_mask, right_mask):
    """
    Finds the optimal horizontal translation of the right mask to maximize
    overlap with the left mask.
    """
    best_score = -1
    best_shift = 0
    height, width = left_mask.shape

    print("Searching for optimal horizontal translation...")
    # Iterate through all possible horizontal shifts
    for shift in range(-width, width):
        # Create translation matrix
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        # Apply the translation
        translated_right_mask = cv2.warpAffine(right_mask, M, (width, height))

        score = dice_coefficient(left_mask, translated_right_mask)

        if score > best_score:
            best_score = score
            best_shift = shift

    return best_shift, best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect and quantify translated features in an image.')
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

        # Find the best translation instead of flipping
        best_shift, best_score = find_best_translation(left_mask, right_mask)
        print(f"Best Dice Similarity Score with translation: {best_score:.4f} at a shift of {best_shift} pixels.")

        # Create the visualization overlay
        print("Creating overlap visualization with optimal translation...")
        original_color_img = cv2.imread(args.image_path)
        img_height, img_width, _ = original_color_img.shape
        left_color_half = original_color_img[:, :img_width//2]

        # Apply the best translation to the right mask
        height, width = right_mask.shape
        M = np.float32([[1, 0, best_shift], [0, 1, 0]])
        translated_right_mask = cv2.warpAffine(right_mask, M, (width, height))
        io.imsave('translated_right_mask.png', translated_right_mask)
        print("Saved translated_right_mask.png")

        # Create a red highlight from the translated mask
        highlight = np.zeros_like(left_color_half)
        highlight[translated_right_mask == 255] = [0, 0, 255]  # Red in BGR

        # Blend the highlight onto the left color image
        overlay = cv2.addWeighted(left_color_half, 1, highlight, 0.8, 0)

        # Save the visualization
        io.imsave('overlap_visualization.png', overlay)
        print("Saved overlap_visualization.png")
