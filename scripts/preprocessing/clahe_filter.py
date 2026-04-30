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
import numpy as np


def preprocess_image(
    image_path, save_steps=False, output_dir="outputs/preprocessing_steps",
    bright_particles=False, norm_min=None, norm_max=None, otsu_threshold=None,
    manual_threshold=None,
    adaptive_threshold=False, adaptive_block_size=51, adaptive_c=15,
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
    bright_particles : bool, optional (default=False)
        If True, skip inversion (use for bright particles on dark backgrounds).
    norm_min, norm_max : int or None, optional (default=None)
        When both are provided, they override the automatic per-image
        min/max used by cv2.normalize(NORM_MINMAX). Use this when
        preprocessing a cropped region of a larger image to keep the
        normalization consistent with the full original (prevents noise
        amplification when extreme-intensity regions like the scale bar
        were cropped out). When either is None, behavior is unchanged.
    otsu_threshold : float or None, optional (default=None)
        When provided, use this as a fixed binary threshold instead of
        running Otsu on the image. Use this to apply a threshold computed
        from the full original image to a cropped region, so both use the
        same intensity cutoff. When None, Otsu runs normally on the
        input image's own histogram (existing behavior).
    manual_threshold : float or None, optional (default=None)
        When provided (via the --threshold CLI flag), use this as a
        fixed binary threshold applied to the LIGHTLY BLURRED ORIGINAL
        image, skipping CLAHE and normalization entirely. This preserves
        the user's intensity-space intuition — the threshold value they
        pass on the CLI is applied directly to the original image pixels.
        Use this for images where Otsu fails, e.g., minority-class dark
        particles. When None, the normal preprocessing pipeline runs.
        Takes precedence over otsu_threshold when both are provided.
    adaptive_threshold : bool, optional (default=False)
        When True (via --threshold adaptive), use OpenCV's adaptive
        Gaussian thresholding instead of Otsu. Each pixel is compared
        against the mean of its local neighborhood (of size
        adaptive_block_size), with adaptive_c subtracted. Good for images
        with uneven lighting or minority-class particles where a single
        global threshold can't capture all particles. CLAHE and
        normalization are skipped (adaptive thresholding is already a
        local method — double-adapting produces noise). Takes precedence
        over both manual_threshold and otsu_threshold.
    adaptive_block_size : int, optional (default=51)
        Size of the local neighborhood used by adaptive thresholding.
        Must be an odd integer >= 3. Larger values average over a wider
        area (less responsive to small features); smaller values are
        more responsive but can be noisier.
    adaptive_c : int, optional (default=15)
        Constant subtracted from the local mean before comparison.
        Larger values make the threshold more conservative (fewer
        pixels are marked foreground); smaller values are more
        permissive (more pixels marked foreground, including more
        noise). Typical range: 5-25.

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

    # Short-circuit: if user provided --threshold adaptive, use OpenCV's
    # adaptive Gaussian thresholding. Skip normalize, CLAHE, and Otsu —
    # adaptive thresholding is ALREADY a local method, so running CLAHE
    # before it double-adapts and amplifies matrix texture noise (verified
    # empirically: CLAHE gave 17,456 noise blobs vs 1,263 real particles
    # on a test image). Only a light blur + adaptiveThreshold + inversion.
    if adaptive_threshold:
        # Light Gaussian blur to reduce JPEG / sensor noise.
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        if save_steps:
            cv2.imwrite(f"{output_dir}/{base_name}_step2_adaptive_blur.png", blurred)
            print(f"Saved: {output_dir}/{base_name}_step2_adaptive_blur.png")

        # cv2.adaptiveThreshold with THRESH_BINARY_INV means:
        # pixels DARKER than (local_mean - C) become foreground.
        # That's the default dark-particle case.
        # For bright particles, use THRESH_BINARY (pixels BRIGHTER than
        # local_mean + C become foreground) — NO post-inversion needed
        # since the polarity is embedded in the threshold type.
        thresh_type = (
            cv2.THRESH_BINARY if bright_particles else cv2.THRESH_BINARY_INV
        )
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type, int(adaptive_block_size), int(adaptive_c)
        )
        if save_steps:
            cv2.imwrite(f"{output_dir}/{base_name}_step3_adaptive_threshold.png", binary)
            print(f"Saved: {output_dir}/{base_name}_step3_adaptive_threshold.png")

        return binary > 0, image

    # Short-circuit: if user provided --threshold VALUE, skip the full
    # normalize → CLAHE → blur pipeline. Instead, apply a light blur to
    # reduce JPEG noise and apply the user's threshold directly to the
    # ORIGINAL image. This preserves intensity-space intuition — the user
    # sees the original image, picks a threshold based on those pixel
    # values, and that's exactly the threshold that gets applied.
    if manual_threshold is not None:
        # Light Gaussian blur to reduce JPEG / sensor noise (same kernel
        # as the regular path's Step 4, but applied to the raw image).
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        if save_steps:
            cv2.imwrite(f"{output_dir}/{base_name}_step2_manual_blur.png", blurred)
            print(f"Saved: {output_dir}/{base_name}_step2_manual_blur.png")

        _, binary = cv2.threshold(
            blurred, float(manual_threshold), 255, cv2.THRESH_BINARY
        )
        if save_steps:
            cv2.imwrite(f"{output_dir}/{base_name}_step3_manual_threshold.png", binary)
            print(f"Saved: {output_dir}/{base_name}_step3_manual_threshold.png")

        # Inversion logic is preserved (same as regular path).
        if not bright_particles:
            binary = 255 - binary
        if save_steps:
            cv2.imwrite(f"{output_dir}/{base_name}_step4_manual_inverted.png", binary)

        return binary > 0, image

    # Step 2: Normalize to 8-bit intensity range (0-255)
    # When anchor values are provided (by interactive-ROI mode), use them so
    # the crop is stretched using the ORIGINAL image's intensity range. This
    # avoids amplifying noise when the crop's own min/max is narrower than
    # the full image's.
    if norm_min is not None and norm_max is not None and norm_max > norm_min:
        # Linear stretch using the external anchor, clipped to [0, 255]
        normalized = np.clip(
            (image.astype(np.float32) - float(norm_min))
            * 255.0 / float(norm_max - norm_min),
            0, 255,
        ).astype(np.uint8)
    else:
        # Original behavior: per-image min/max stretch
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

    # Step 5: Binarize. Either use a provided threshold (from the full
    # original image — interactive-ROI mode) or let Otsu pick one from the
    # current image's histogram (default behavior).
    if otsu_threshold is not None:
        _, binary = cv2.threshold(
            blurred, float(otsu_threshold), 255, cv2.THRESH_BINARY
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

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

def compute_full_image_otsu(image_path, norm_min=None, norm_max=None):
    """
    Run the same normalize → CLAHE → blur → Otsu sequence that
    preprocess_image uses, but only to extract the Otsu threshold value.

    Used by interactive-ROI mode to compute a threshold from the ORIGINAL
    full image, which is then passed back into preprocess_image(crop) so
    the crop uses the same intensity cutoff as the full image would have.

    Parameters
    ----------
    image_path : str
        Path to the original full image.
    norm_min, norm_max : int or None
        Optional anchor values for normalization. Should typically match
        whatever crop_to_cache computed (the full image's min/max), but
        since this IS the full image, passing them in doesn't change
        anything — per-image min/max would be identical. Kept for
        symmetry with preprocess_image's signature.

    Returns
    -------
    float or None
        Otsu threshold value in [0, 255]. Returns None if the image
        cannot be read.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # Mirror preprocess_image's steps 2-4 exactly
    if norm_min is not None and norm_max is not None and norm_max > norm_min:
        normalized = np.clip(
            (image.astype(np.float32) - float(norm_min))
            * 255.0 / float(norm_max - norm_min),
            0, 255,
        ).astype(np.uint8)
    else:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    thresh_val, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return float(thresh_val)
