"""
Allsky Mean Target Controller for automatic gain adjustment
Based on AllSky project's mean target algorithm
https://github.com/AllskyTeam/allsky
"""

import numpy as np
import cv2


class AllskyMeanController:
    """
    Controller for automatic gain adjustment based on mean brightness target

    Analyzes the mean brightness of a central ROI and adjusts gain to reach
    a target brightness value over time.

    Usage:
        controller = AllskyMeanController(mean_target=0.30, mean_threshold=0.05, max_gain=200)

        # After capturing an image
        measured_mean = controller.calculate_mean('/path/to/image.jpg')
        new_gain = controller.update(current_gain, measured_mean)

        # Use new_gain for next capture
    """

    def __init__(self, mean_target=0.30, mean_threshold=0.05, max_gain=200):
        """
        Initialize the Mean Target controller

        Args:
            mean_target (float): Target mean brightness (0.0-1.0)
            mean_threshold (float): Tolerance around target before adjusting
            max_gain (int): Maximum gain allowed
        """
        self.mean_target = mean_target
        self.mean_threshold = mean_threshold
        self.max_gain = max_gain
        self.min_gain = 1
        self.current_gain = None

        # History for smoothing (optional, can help reduce oscillations)
        self.mean_history = []
        self.history_size = 3  # Keep last 3 measurements

    def calculate_mean(self, image_path):
        """
        Calculate mean brightness from central ROI of image

        Args:
            image_path (str): Path to JPEG image file

        Returns:
            float: Mean brightness value (0.0-1.0), or None on error
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"[AllskyMeanController] Failed to load image: {image_path}")
                return None

            # Convert to grayscale for brightness analysis
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Define central ROI (middle 50% of image)
            h, w = gray.shape
            roi_h = int(h * 0.5)
            roi_w = int(w * 0.5)
            y_start = int((h - roi_h) / 2)
            x_start = int((w - roi_w) / 2)

            roi = gray[y_start:y_start+roi_h, x_start:x_start+roi_w]

            # Calculate mean and normalize to 0.0-1.0
            mean_value = np.mean(roi) / 255.0

            return mean_value

        except Exception as e:
            print(f"[AllskyMeanController] Error calculating mean: {e}")
            return None

    def update(self, current_gain, measured_mean):
        """
        Calculate new gain based on measured mean brightness

        Args:
            current_gain (int/float): Current gain value
            measured_mean (float): Measured mean brightness (0.0-1.0)

        Returns:
            int: New gain value to use for next capture
        """
        if measured_mean is None:
            return int(current_gain)

        self.current_gain = current_gain

        # Add to history for smoothing
        self.mean_history.append(measured_mean)
        if len(self.mean_history) > self.history_size:
            self.mean_history.pop(0)

        # Use smoothed mean if we have enough history
        if len(self.mean_history) >= 2:
            smoothed_mean = np.mean(self.mean_history)
        else:
            smoothed_mean = measured_mean

        # Calculate deviation from target
        deviation = self.mean_target - smoothed_mean

        # Check if we're within threshold
        if abs(deviation) < self.mean_threshold:
            # Within tolerance - no adjustment needed
            return int(current_gain)

        # Calculate gain adjustment
        # Use proportional controller: larger deviation = larger adjustment
        # Factor of 2.0 provides good responsiveness without overshooting
        if smoothed_mean > 0.01:  # Avoid division by very small numbers
            adjustment_factor = 1.0 + (deviation * 2.0)
            # Limit adjustment to prevent large jumps
            adjustment_factor = np.clip(adjustment_factor, 0.5, 2.0)
        else:
            # Image is very dark - increase gain significantly
            adjustment_factor = 2.0

        new_gain = current_gain * adjustment_factor

        # Clamp to valid range (min_gain to max_gain)
        new_gain = int(np.clip(new_gain, self.min_gain, self.max_gain))

        return new_gain

    def reset_history(self):
        """Reset the mean history (useful when starting a new timelapse)"""
        self.mean_history = []


def stack_images_simple(image_paths, output_path, quality=95):
    """
    Stack multiple images using simple mean averaging (no alignment).
    Suitable for Allsky timelapse where camera is fixed.

    Args:
        image_paths (list): List of paths to JPEG images to stack
        output_path (str): Path for the output stacked JPEG
        quality (int): JPEG quality (1-100)

    Returns:
        bool: True if successful, False otherwise
    """
    if not image_paths:
        print("[AllskyStack] No images to stack")
        return False

    if len(image_paths) == 1:
        # Only one image, just copy it
        try:
            import shutil
            shutil.copy2(image_paths[0], output_path)
            return True
        except Exception as e:
            print(f"[AllskyStack] Error copying single image: {e}")
            return False

    try:
        # Load first image to get dimensions
        first_img = cv2.imread(image_paths[0])
        if first_img is None:
            print(f"[AllskyStack] Failed to load first image: {image_paths[0]}")
            return False

        # Initialize accumulator as float32 for precision
        accumulator = first_img.astype(np.float32)
        valid_count = 1

        # Add remaining images
        for path in image_paths[1:]:
            img = cv2.imread(path)
            if img is not None and img.shape == first_img.shape:
                accumulator += img.astype(np.float32)
                valid_count += 1
            else:
                print(f"[AllskyStack] Skipping invalid/incompatible image: {path}")

        if valid_count == 0:
            print("[AllskyStack] No valid images to stack")
            return False

        # Calculate mean
        stacked = (accumulator / valid_count).astype(np.uint8)

        # Save result
        cv2.imwrite(output_path, stacked, [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"[AllskyStack] Stacked {valid_count} images -> {output_path}")
        return True

    except Exception as e:
        print(f"[AllskyStack] Error stacking images: {e}")
        return False


def stack_timelapse_images(pic_dir, timestamp, stack_count, quality=95):
    """
    Stack timelapse images in groups of stack_count.
    Creates new stacked images and returns the list of stacked image paths.

    Args:
        pic_dir (str): Directory containing the images
        timestamp (str): Timestamp prefix for the timelapse images
        stack_count (int): Number of images per stack
        quality (int): JPEG quality for output images

    Returns:
        tuple: (list of stacked image paths, list of original images to delete)
    """
    import glob
    import os

    # Find all timelapse images with this timestamp
    pattern = os.path.join(pic_dir, f"{timestamp}_*.jpg")
    all_images = sorted(glob.glob(pattern))

    if not all_images:
        print(f"[AllskyStack] No images found matching {pattern}")
        return [], []

    print(f"[AllskyStack] Found {len(all_images)} images to process with stack_count={stack_count}")

    stacked_images = []
    original_images = []
    stack_index = 0

    # Process images in groups of stack_count
    for i in range(0, len(all_images), stack_count):
        group = all_images[i:i + stack_count]

        if len(group) < stack_count:
            # Last group is incomplete - still stack what we have
            if len(group) == 0:
                continue
            print(f"[AllskyStack] Last group has only {len(group)} images (expected {stack_count})")

        # Create output filename for this stacked image
        output_path = os.path.join(pic_dir, f"{timestamp}_stacked_{stack_index:04d}.jpg")

        if stack_images_simple(group, output_path, quality):
            stacked_images.append(output_path)
            original_images.extend(group)
            stack_index += 1
        else:
            print(f"[AllskyStack] Failed to stack group {stack_index}")
            # Keep original images in case of failure
            stacked_images.extend(group)

    print(f"[AllskyStack] Created {len(stacked_images)} stacked images from {len(original_images)} originals")
    return stacked_images, original_images
