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

# Import required libraries
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
# from skimage.draw import ellipse_perimeter

CONTOUR_THICKNESS = 5  # Thickness of contour lines when drawing


def measure_particles(
    regions,
    labeled_image,
    original_image,
    nm_per_pixel,
    image_path,
    min_size_px=5,
    max_size_px=None,
    only_morphology=None,
    # Morphology classification thresholds
    spherical_ar_max=1.5,
    spherical_c_min=0.75,
    spherical_s_min=0.90,
    aggregate_s_max=0.85,
    aggregate_c_max=0.60,
    rodlike_ar_min=1.8,
    rodlike_s_min=0.80,
):
    """
    Measures the diameters of segmented nanoparticles and overlays contours on the original image.
    Also classify particle morphology into three types: spherical, road-like, and aggregate.

    Parameters:
    -----------
    regions : list of skimage.measure._regionprops.RegionProperties
        Regions obtained from segmenting the binary image.
    labeled_image : np.ndarray
        Labeled mask of the segmented particles.
    original_image : np.ndarray
        Original grayscale image used for overlay and visualization.
    nm_per_pixel : float
        Calibration factor to convert pixel measurements to nanometers.
    min_diam_nm : float, optional
        Minimum physical diameter (in nanometers) to be considered valid.

    Returns:
    --------
    results : list of float
        Measured diameters (in nanometers) of valid particles.
    """

    # Reload original image
    img_for_overlay = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert to BGR
    img_for_overlay = cv2.cvtColor(img_for_overlay, cv2.COLOR_GRAY2BGR)

    # Use for all contour overlays
    true_contour_img = img_for_overlay.copy()
    circular_img = true_contour_img.copy()
    elliptical_img = true_contour_img.copy()
    combined_img = true_contour_img.copy()
    true_circular_img = true_contour_img.copy()
    morphology_overlay = img_for_overlay.copy()

    # Initialize list to store particle diameters (in nanometers)
    diameters_pixels = []
    diameters_nm = []
    centroids = []

    # Convert pixel-based size thresholds to area (scale-independent)
    # A = π * (d/2)^2, where d is the diameter in pixels
    min_area_px = np.pi * (min_size_px / 2) ** 2

    # Max size is optional - if not specified, use infinity
    if max_size_px is not None:
        max_area_px = np.pi * (max_size_px / 2) ** 2
    else:
        max_area_px = float("inf")  # No upper limit

    # Iterate over each detected region (particle)
    for region in regions:
        # Filter by area (safety net - already filtered in segmentation)
        if min_area_px <= region.area <= max_area_px:
            # Equivalent diameter of a region (diameter of a circle with same area)
            d_px = region.equivalent_diameter

            # Convert diameter from pixels to nanometers
            d_nm = d_px * nm_per_pixel

            # Get the centroid coordinates (center of the particle)
            y, x = region.centroid

            # Create a binary mask for the current region
            # 'labeled == region.label' will be True for pixels belonging to this region
            # Convert boolean mask to uint8 (0 or 1) so OpenCV can process it
            
            region_mask = (labeled_image == region.label).astype(np.uint8)

            # --- Classify morphology FIRST ---
            kernel_size = max(3, int(np.sqrt(region.area) * 0.1) // 2 * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            smoothed_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
            smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_OPEN, kernel)

            smooth_contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if smooth_contours:
                smooth_cnt = max(smooth_contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(smooth_cnt, True)
                smooth_area = cv2.contourArea(smooth_cnt)
            else:
                perimeter = region.perimeter
                smooth_area = region.area

            if len(region.coords) >= 5:
                major_axis = region.major_axis_length
                minor_axis = region.minor_axis_length
                aspect_ratio = major_axis / max(minor_axis, 1e-6)
                circularity = (4 * np.pi * smooth_area) / max(perimeter**2, 1e-6)
                if smooth_contours:
                    hull = cv2.convexHull(smooth_cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = smooth_area / max(hull_area, 1e-6)
                else:
                    solidity = region.solidity
                extent = region.extent

                if solidity < aggregate_s_max or circularity < aggregate_c_max:
                    morphology = "aggregate"
                elif (
                    aspect_ratio < spherical_ar_max
                    and circularity > spherical_c_min
                    and solidity > spherical_s_min
                ):
                    morphology = "spherical"
                elif aspect_ratio >= rodlike_ar_min and solidity > rodlike_s_min:
                    morphology = "rod-like"
                else:
                    morphology = "aggregate"
            else:
                aspect_ratio = 1.0
                circularity = 1.0
                solidity = 1.0
                extent = 1.0
                morphology = "aggregate"

            # --- Skip if doesn't match filter ---
            if only_morphology is not None and morphology != only_morphology:
                continue

            # --- 1. True Contour (in BLUE) ---
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(true_contour_img, contours, -1, (255, 0, 0), CONTOUR_THICKNESS)
            cv2.drawContours(combined_img, contours, -1, (255, 0, 0), CONTOUR_THICKNESS)
            cv2.drawContours(true_circular_img, contours, -1, (255, 0, 0), CONTOUR_THICKNESS)

            # --- Morphology color contour (inline — avoids index desync with --only-morphology) ---
            morph_color = {
                "spherical": (255, 0, 0),   # Blue (BGR)
                "rod-like": (255, 255, 0),  # Bright Cyan
                "aggregate": (255, 0, 255), # Magenta
            }
            cv2.drawContours(
                morphology_overlay, contours, -1,
                morph_color.get(morphology, (255, 255, 255)), CONTOUR_THICKNESS
            )

            # --- 2. Circular Equivalent Contour (in CYAN) ---
            d_px = region.equivalent_diameter
            y, x = region.centroid
            cv2.circle(circular_img, (int(x), int(y)), int(d_px / 2), (255, 0, 255), CONTOUR_THICKNESS)
            cv2.circle(combined_img, (int(x), int(y)), int(d_px / 2), (255, 0, 255), CONTOUR_THICKNESS)
            cv2.circle(true_circular_img, (int(x), int(y)), int(d_px / 2), (255, 0, 255), CONTOUR_THICKNESS)

            # --- 3. Elliptical Equivalent Contour (in PINK) ---
            # cv2.fitEllipse is an ill-conditioned least-squares fit on
            # contours that are essentially collinear (1-pixel-wide strips,
            # edge fragments). It can return degenerate ellipses with
            # near-zero minor axes or absurdly large major axes (e.g.,
            # ~7.7e7 px), which render as horizontal or vertical lines
            # across the entire overlay. Reject these by requiring:
            #   - both axes >= 1 pixel (reject near-zero)
            #   - both axes <= 2x the contour's bounding-box diagonal
            #     (reject axis-length explosions)
            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, angle) = ellipse
                    x_bb, y_bb, cw_bb, ch_bb = cv2.boundingRect(contour)
                    bbox_diag = (cw_bb * cw_bb + ch_bb * ch_bb) ** 0.5
                    max_axis = max(10.0, bbox_diag * 2.0)
                    if (axes[0] >= 1.0 and axes[1] >= 1.0
                            and axes[0] <= max_axis and axes[1] <= max_axis):
                        cv2.ellipse(
                            elliptical_img, ellipse, (255, 0, 255), CONTOUR_THICKNESS
                        )
                        cv2.ellipse(
                            combined_img, ellipse, (255, 0, 255), CONTOUR_THICKNESS
                        )

            # --- Store measurements ---
            diameters_pixels.append(d_px)
            diameters_nm.append(d_nm)
            centroids.append(
                {
                    "y": region.centroid[0],
                    "x": region.centroid[1],
                    "aspect_ratio": aspect_ratio,
                    "circularity": circularity,
                    "solidity": solidity,
                    "extent": extent,
                    "morphology": morphology,
                }
            )

    # Save images using original filename + suffix, preserving original extension
    os.makedirs("outputs/figures", exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]  # e.g., "SEM_Sample_Image"
    ext = os.path.splitext(image_path)[1]  # e.g., ".tif"

    # Scale legend text to image size (reference: 2048px width)
    _font_scale = max(0.5, img_for_overlay.shape[1] / 700)
    _font_thick = max(1, int(_font_scale * 2.5))
    _legend_gap = max(30, int(img_for_overlay.shape[1] / 20))

    true_path = f"outputs/figures/{stem}_true_contours{ext}"
    circ_path = f"outputs/figures/{stem}_circular_equivalent{ext}"
    ell_path = f"outputs/figures/{stem}_elliptical_equivalent{ext}"
    all_path = f"outputs/figures/{stem}_all_contour_types{ext}"

    cv2.imwrite(true_path, true_contour_img)
    cv2.imwrite(circ_path, circular_img)
    cv2.imwrite(ell_path, elliptical_img)
    cv2.imwrite(all_path, combined_img)

    print("Saved all contour types:")
    print(" -", true_path)
    print(" -", circ_path)
    print(" -", ell_path)
    print(" -", all_path)

    # True contour + circular equivalent combined overlay with legend
    tc_path = f"outputs/figures/{stem}_true_circular{ext}"
    legend_y = _legend_gap
    legend_items_tc = [
        ("True Contour", (255, 0, 0)),
        ("Circular Equivalent", (255, 0, 255)),
    ]
    for text, color in legend_items_tc:
        cv2.putText(
            true_circular_img, text, (15, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, _font_scale, color, _font_thick,
        )
        legend_y += _legend_gap
    cv2.imwrite(tc_path, true_circular_img)
    print(" -", tc_path)

    # Morphology distribution summary
    morph_types = [c["morphology"] for c in centroids]
    from collections import Counter
    counts = Counter(morph_types)
    print(f"Morphology distribution: {dict(counts)}")

    # # Morphology overlay
    # # True contour + circular equivalent combined overlay with legend
    # tc_path = f"outputs/figures/{stem}_true_circular{ext}"
    # legend_y = _legend_gap
    # for text, color in [("True Contour", (255, 0, 0)), ("Circular Equivalent", (0, 255, 255))]:
    #     cv2.putText(
    #         true_circular_img, text, (15, legend_y),
    #         cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 8,
    #     )
    #     legend_y += _legend_gap
    # cv2.imwrite(tc_path, true_circular_img)
    # print(" -", tc_path)

    # # Morphology distribution summary
    # morph_types = [c["morphology"] for c in centroids]
    # from collections import Counter
    # counts = Counter(morph_types)
    # print(f"Morphology distribution: {dict(counts)}")

    # Add legend to morphology overlay (only for morphologies present)
    legend_items = [
        ("Spherical", (255, 0, 0), "spherical"),
        ("Rod-like", (255, 255, 0), "rod-like"),
        ("Aggregate", (255, 0, 255), "aggregate"),
    ]
    legend_y = _legend_gap
    for text, color, morph in legend_items:
        if morph in morph_types:
            cv2.putText(
                morphology_overlay,
                text,
                (15, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                _font_scale,
                color,
                _font_thick,
            )
            legend_y += _legend_gap

    # Save morphology overlay
    morph_path = f"outputs/figures/{stem}_morphology_overlay{ext}"
    cv2.imwrite(morph_path, morphology_overlay)
    print(" -", morph_path)

    # Convert to DataFrame with ALL metrics
    df = pd.DataFrame(
        {
            "Diameter (nm)": diameters_nm,
            "Diameter (pixels)": diameters_pixels,
            "Centroid_X": [c["x"] for c in centroids],
            "Centroid_Y": [c["y"] for c in centroids],
            "Aspect_Ratio": [c["aspect_ratio"] for c in centroids],
            "Circularity": [c["circularity"] for c in centroids],
            "Solidity": [c["solidity"] for c in centroids],
            "Extent": [c["extent"] for c in centroids],
            "Morphology": [c["morphology"] for c in centroids],
        }
    )
    df.to_csv(f"outputs/results/{stem}_nanoparticle_data.csv", index=False)

    # Return the list of diameters
    return diameters_nm, combined_img, df


def export_to_latex(diameters, img_path, out_path="outputs/report.tex"):
    """
    Export basic stats to a tiny LaTeX snippet. Safe if diameters is empty.
    """
    import os
    from scipy.stats import describe

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base = os.path.basename(img_path)

    with open(out_path, "a") as f:
        f.write("\n\\section*{%s}\n" % base)

        if not diameters:
            # No crash; just record that nothing passed filters
            f.write("\\textit{No particles detected after filtering.}\n")
            f.write("\\begin{tabular}{ll}\nCount & 0 \\\\\n\\end{tabular}\n")
            return

        stats = describe(diameters)
        n = stats.nobs
        mean = stats.mean
        var = stats.variance if stats.variance is not None else 0.0
        mn, mx = stats.minmax

        f.write("\\begin{tabular}{ll}\n")
        f.write(f"Count & {n} \\\\\n")
        f.write(f"Mean (nm) & {mean:.2f} \\\\\n")
        f.write(f"Variance (nm$^2$) & {var:.2f} \\\\\n")
        f.write(f"Min/Max (nm) & {mn:.2f} / {mx:.2f} \\\\\n")
        f.write("\\end{tabular}\n")


def export_summary_csv(diameters, df, img_path):
    """
    Export summary statistics to CSV format for SINGLE IMAGE analysis.

    This function is called ONLY in single-image mode to generate a per-image
    summary CSV file. In batch mode, the batch_summary.csv is generated instead.

    Parameters:
    -----------
    diameters : list of float
        Particle diameters in nanometers
    df : pd.DataFrame
        Full particle data with morphology classifications
    img_path : str
        Path to the analyzed image

    Returns:
    --------
    None
        CSV file written to outputs/results/{image_name}_summary.csv
    """
    import os
    import pandas as pd
    from scipy.stats import describe

    # Generate output filename from image name
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = f"outputs/results/{base_name}_summary.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Handle empty results gracefully
    if not diameters or len(diameters) == 0:
        summary_data = {
            "Image": [os.path.basename(img_path)],
            "Total_Particles": [0],
            "Mean_Diameter_nm": [0.0],
            "Std_Diameter_nm": [0.0],
            "Median_Diameter_nm": [0.0],
            "Min_Diameter_nm": [0.0],
            "Max_Diameter_nm": [0.0],
            "Spherical_Count": [0],
            "RodLike_Count": [0],
            "Aggregate_Count": [0],
        }
    else:
        # Calculate descriptive statistics
        stats = describe(diameters)

        # Count morphology types
        if df is not None and "Morphology" in df.columns:
            spherical_count = len(df[df["Morphology"] == "spherical"])
            rodlike_count = len(df[df["Morphology"] == "rod-like"])
            aggregate_count = len(df[df["Morphology"] == "aggregate"])
        else:
            spherical_count = 0
            rodlike_count = 0
            aggregate_count = 0

        # Create summary dictionary
        summary_data = {
            "Image": [os.path.basename(img_path)],
            "Total_Particles": [int(stats.nobs)],
            "Mean_Diameter_nm": [float(stats.mean)],
            "Std_Diameter_nm": [float(stats.variance**0.5 if stats.variance else 0.0)],
            "Median_Diameter_nm": [float(pd.Series(diameters).median())],
            "Min_Diameter_nm": [float(stats.minmax[0])],
            "Max_Diameter_nm": [float(stats.minmax[1])],
            "Spherical_Count": [int(spherical_count)],
            "RodLike_Count": [int(rodlike_count)],
            "Aggregate_Count": [int(aggregate_count)],
        }

    # Convert to DataFrame and save
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(out_path, index=False)

    print(f"Saved summary statistics: {out_path}")
