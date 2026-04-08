# SPDX-License-Identifier: GPL-3.0-or-later
#
# NanoPSD: Automated Nanoparticle Shape Distribution Analysis
# Copyright (C) 2026 Md Fazlul Huq
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Import OpenCV for image processing functions
import cv2
import os


def preprocess_image(
    image_path, save_steps=False, output_dir="outputs/preprocessing_steps", bright_particles=False
):
    """
    Preprocesses a microscopy image by enhancing contrast, smoothing, and thresholding.

    Parameters:
    -----------
    image_path : str
        Path to the grayscale input image.
    save_steps : bool, optional (default=False)
        If True, save intermediate preprocessing steps for visualization.
    output_dir : str, optional (default="outputs/preprocessing_steps")
        Directory to save intermediate images.

    Returns:
    --------
    binary : np.ndarray (bool)
        A binary image (True for foreground/particles, False for background).
    image : np.ndarray (uint8)
        The original grayscale image.
    """

    # Create output directory if saving steps
    if save_steps:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Step 1: Read the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if save_steps:
        cv2.imwrite(f"{output_dir}/{base_name}_step1_original.png", image)
        print(f"Saved: {output_dir}/{base_name}_step1_original.png")

    # Step 2: Normalize to 8-bit intensity range (0-255)
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    if save_steps:
        cv2.imwrite(f"{output_dir}/{base_name}_step2_normalized.png", normalized)
        print(f"Saved: {output_dir}/{base_name}_step2_normalized.png")

    # Step 3: Apply CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    if save_steps:
        cv2.imwrite(f"{output_dir}/{base_name}_step3_clahe.png", enhanced)
        print(f"Saved: {output_dir}/{base_name}_step3_clahe.png")

    # Step 4: Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    if save_steps:
        cv2.imwrite(f"{output_dir}/{base_name}_step4_gaussian_blur.png", blurred)
        print(f"Saved: {output_dir}/{base_name}_step4_gaussian_blur.png")

    # Step 5: Apply Otsu's thresholding to binarize the image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if save_steps:
        cv2.imwrite(f"{output_dir}/{base_name}_step5_otsu_threshold.png", binary)
        print(f"Saved: {output_dir}/{base_name}_step5_otsu_threshold.png")

    # Step 6: Invert the binary image (skip if bright particles)
    if not bright_particles:
        binary = 255 - binary

    if save_steps:
        cv2.imwrite(f"{output_dir}/{base_name}_step6_inverted.png", binary)
        print(f"Saved: {output_dir}/{base_name}_step6_inverted.png")

    # Return the binary image as a boolean array and the original normalized image
    return binary > 0, image
