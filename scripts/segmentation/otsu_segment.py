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

from skimage import (
    measure,
    morphology,
)  # Import particle measurement and morphological tools from scikit-image
from scipy import ndimage  # Import functions for binary image manipulation from SciPy
import numpy as np
import cv2
import os


def segment_particles(
    binary_image,
    min_size=3,
    max_size=None,
    save_steps=False,
    output_dir="outputs/segmentation_steps",
    image_name="image",
):
    """
    Segments particles in a binary image using connected component analysis,
    removes small and large noise objects, fills internal holes, and returns labeled regions.

    NOTE: Scale bar exclusion should be done BEFORE calling this function
    by masking the binary_image (setting excluded pixels to False).

    Parameters:
    -----------
    binary_image : np.ndarray (bool)
        Binary mask of the image where True = foreground (particle), False = background.
    min_size : int, optional (default=3)
        Minimum size (in pixels) for objects to be retained. Smaller regions are considered noise.
    max_size : int, optional (default=None)
        Maximum size (in pixels) for objects to be retained. Larger regions are filtered out.
        If None, no maximum filter is applied.

    Returns:
    --------
    labeled : np.ndarray (int)
        Labeled image where each connected component (particle) has a unique positive integer label.
        Background is labeled as 0.
    regions : list of skimage.measure._regionprops.RegionProperties
        A list of region properties for each labeled particle, which can be used for size, shape, etc.
    """

    # Setup for saving intermediate steps
    if save_steps:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: Save input binary mask
    if save_steps:
        input_vis = (binary_image * 255).astype(np.uint8)
        cv2.imwrite(f"{output_dir}/{image_name}_step1_input_binary.png", input_vis)
        print(f"Saved: {output_dir}/{image_name}_step1_input_binary.png")

    # Step 2: Remove small objects from the binary image
    # This is a morphological operation to eliminate noise and tiny specks
    # Any connected group of True pixels smaller than min_size is removed
    cleaned = morphology.remove_small_objects(binary_image, min_size=min_size)

    if save_steps:
        cleaned_vis = (cleaned * 255).astype(np.uint8)
        cv2.imwrite(
            f"{output_dir}/{image_name}_step2_after_small_removal.png", cleaned_vis
        )
        print(f"Saved: {output_dir}/{image_name}_step2_after_small_removal.png")

    # Step 3: Remove large objects if max_size is specified
    if max_size is not None:
        # Label to get regions, then filter by area
        temp_labeled, _ = ndimage.label(cleaned)
        temp_regions = measure.regionprops(temp_labeled)

        # Create mask excluding large objects
        mask = np.ones_like(cleaned, dtype=bool)
        for region in temp_regions:
            if region.area > max_size:
                mask[temp_labeled == region.label] = False

        cleaned = cleaned & mask

    if save_steps:
        cleaned_vis = (cleaned * 255).astype(np.uint8)
        cv2.imwrite(
            f"{output_dir}/{image_name}_step3_after_large_removal.png", cleaned_vis
        )
        print(f"Saved: {output_dir}/{image_name}_step3_after_large_removal.png")

    # Step 4: Fill internal holes within objects (particles)
    # Ensures that ring-shaped particles or porous features are treated as solid
    # For example, a particle shaped like a donut will be filled in
    filled = ndimage.binary_fill_holes(cleaned)

    if save_steps:
        filled_vis = (filled * 255).astype(np.uint8)
        cv2.imwrite(
            f"{output_dir}/{image_name}_step4_after_hole_filling.png", filled_vis
        )
        print(f"Saved: {output_dir}/{image_name}_step4_after_hole_filling.png")

    # Step 5: Label connected components in the binary image
    # Each group of connected True pixels gets a unique label: 1, 2, 3, ..., N
    # Background pixels (False) are labeled as 0
    # '_' represents `num_particles` counts the number of detected objects
    labeled, _ = ndimage.label(filled)

    if save_steps:
        # Create colorized visualization of labeled components
        # Normalize labels to 0-255 range for visualization
        if labeled.max() > 0:
            labeled_vis = ((labeled / labeled.max()) * 255).astype(np.uint8)
            # Apply colormap for better visualization
            labeled_color = cv2.applyColorMap(labeled_vis, cv2.COLORMAP_JET)
            # Set background (label 0) to black
            labeled_color[labeled == 0] = [0, 0, 0]
            cv2.imwrite(
                f"{output_dir}/{image_name}_step5_labeled_components.png", labeled_color
            )
            print(f"Saved: {output_dir}/{image_name}_step5_labeled_components.png")
        else:
            # No particles detected - save blank image
            blank = np.zeros((*labeled.shape, 3), dtype=np.uint8)
            cv2.imwrite(
                f"{output_dir}/{image_name}_step5_labeled_components.png", blank
            )
            print(
                f"Saved: {output_dir}/{image_name}_step5_labeled_components.png (no particles)"
            )

    # Step 6: Compute region properties of the labeled image
    # This returns measurements like area, centroid, bounding box, etc., for each particle
    regions = measure.regionprops(labeled)

    # Step 7: Return the labeled image and list of particle properties
    return labeled, regions
