
import numpy as np
import cv2
import argparse
from create_mask import create_mask

def dice_coefficient(mask1, mask2):
    """Calculates the Dice similarity coefficient between two binary masks."""
    # Ensure masks are boolean
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)

    if total == 0:
        return 1.0  # Both masks are empty, perfect match

    return 2. * intersection / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare reflections in an image by flipping and comparing masks.')
    parser.add_argument('image_path', type=str, help='The path to the input image.')
    args = parser.parse_args()

    # Generate left and right masks
    left_mask = create_mask(args.image_path, 'left')
    right_mask = create_mask(args.image_path, 'right')

    if left_mask is not None and right_mask is not None:
        # Flip the left mask horizontally
        flipped_left_mask = cv2.flip(left_mask, 1)

        # Calculate the Dice similarity coefficient
        dice_score = dice_coefficient(flipped_left_mask, right_mask)

        print(f"Dice Similarity Score: {dice_score:.4f}")
