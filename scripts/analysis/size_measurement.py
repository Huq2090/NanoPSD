# Import required libraries
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.draw import ellipse_perimeter

CONTOUR_THICKNESS = 1  # Thickness of contour lines when drawing


def measure_particles(
    regions,
    labeled_image,
    original_image,
    nm_per_pixel,
    image_path,
    min_diam_nm=5,
    max_diam_nm=35,
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

    # Convert grayscale image to BGR (3-channel) so we can draw colored contours
    true_contour_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    circular_img = true_contour_img.copy()
    elliptical_img = true_contour_img.copy()
    combined_img = true_contour_img.copy()

    # Initialize list to store particle diameters (in nanometers)
    diameters_pixels = []
    diameters_nm = []
    centroids = []

    # Convert the minimum valid diameter to area in pixels using the circle area formula
    # A = π * (d/2)^2, where d is the minimum diameter in pixels
    min_area_px = np.pi * ((min_diam_nm / nm_per_pixel) / 2) ** 2
    max_area_px = np.pi * ((max_diam_nm / nm_per_pixel) / 2) ** 2

    # Iterate over each detected region (particle)
    for region in regions:
        # Only consider regions with area >= minimum threshold
        if max_area_px >= region.area >= min_area_px:
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

            # --- 1. True Contour (in BLUE) ---
            # Find contours in the binary mask
            # Since we're looking at one particle at a time, there should typically be just one contour
            # cv2.RETR_EXTERNAL: retrieve only outer contours (ignores internal holes)
            # cv2.CHAIN_APPROX_SIMPLE: compresses contour points (saves memory)
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Draw the contours on the image
            # -1 indicates all contours found
            # (0, 0, 255) sets the contour color to red (in BGR format)
            # Thickness of 1 pixel
            cv2.drawContours(
                true_contour_img, contours, -1, (255, 0, 0), CONTOUR_THICKNESS
            )
            cv2.drawContours(combined_img, contours, -1, (255, 0, 0), CONTOUR_THICKNESS)

            # --- 2. Circular Equivalent Contour (in RED) ---
            d_px = region.equivalent_diameter
            y, x = region.centroid
            rr, cc = ellipse_perimeter(int(y), int(x), int(d_px / 2), int(d_px / 2))
            rr = np.clip(rr, 0, original_image.shape[0] - 1)
            cc = np.clip(cc, 0, original_image.shape[1] - 1)
            circular_img[rr, cc] = (0, 0, 255)
            combined_img[rr, cc] = (0, 0, 255)

            # --- 3. Elliptical Equivalent Contour (in PINK) ---
            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(
                        elliptical_img, ellipse, (255, 0, 255), CONTOUR_THICKNESS
                    )
                    cv2.ellipse(combined_img, ellipse, (255, 0, 255), CONTOUR_THICKNESS)

            # Morphology Classification
            # Calculate shape metrics
            perimeter = region.perimeter
            if len(region.coords) >= 5:
                # Get perimeter (already calculated earlier but need it here too)
                perimeter = region.perimeter

                major_axis = region.major_axis_length
                minor_axis = region.minor_axis_length
                aspect_ratio = major_axis / max(minor_axis, 1e-6)

                circularity = (4 * np.pi * region.area) / max(perimeter**2, 1e-6)
                solidity = region.solidity
                extent = region.extent

                # Classification logic (priority: aggregate > spherical > rod-like)
                if solidity < 0.85 or circularity < 0.60:
                    morphology = "aggregate"
                elif aspect_ratio < 1.5 and circularity > 0.75 and solidity > 0.90:
                    morphology = "spherical"
                elif aspect_ratio >= 1.8 and solidity > 0.80:
                    morphology = "rod-like"
                else:
                    morphology = "aggregate"
            else:
                aspect_ratio = 1.0
                circularity = 1.0
                solidity = 1.0
                extent = 1.0
                morphology = "aggregate"

            # Add the diameter to the result list
            # Store all measurements in lists
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

    # Morphology overlay
    morphology_overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    color_map = {
        "spherical": (0, 255, 0),  # Green
        "rod-like": (255, 0, 0),  # Blue (BGR)
        "aggregate": (0, 0, 255),  # Red
    }

    # Debug counts
    morph_types = [c["morphology"] for c in centroids]
    from collections import Counter

    counts = Counter(morph_types)
    print(f"Morphology distribution: {dict(counts)}")

    region_idx = 0
    for region in regions:
        if max_area_px >= region.area >= min_area_px:
            if region_idx >= len(centroids):
                break

            morph = centroids[region_idx]["morphology"]
            color = color_map.get(morph, (255, 255, 255))

            region_mask = (labeled_image == region.label).astype(np.uint8)
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(morphology_overlay, contours, -1, color, CONTOUR_THICKNESS)

            cy, cx = region.centroid
            label = morph[0].upper()
            cv2.putText(
                morphology_overlay,
                label,
                (int(cx) - 5, int(cy) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            region_idx += 1

    # Add legend
    legend_y = 30
    cv2.putText(
        morphology_overlay,
        "Green = Spherical",
        (10, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        morphology_overlay,
        "Blue = Rod-like",
        (10, legend_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        morphology_overlay,
        "Red = Aggregate",
        (10, legend_y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

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
    df.to_csv("outputs/results/nanoparticle_data.csv", index=False)

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
