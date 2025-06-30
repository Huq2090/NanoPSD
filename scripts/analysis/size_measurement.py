# Import required libraries
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.draw import ellipse_perimeter

def measure_particles(regions, labeled_image, original_image, nm_per_pixel, min_diam_nm=5, max_diam_nm=35):
    """
    Measures the diameters of segmented nanoparticles and overlays contours on the original image.

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
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw the contours on the image
            # -1 indicates all contours found
            # (255, 0, 0) sets the contour color to blue (in BGR format)
            # Thickness of 1 pixel
            cv2.drawContours(true_contour_img, contours, -1, (255, 0, 0), 1)
            cv2.drawContours(combined_img, contours, -1, (255, 0, 0), 1)

            # --- 2. Circular Equivalent Contour (in GREEN) ---
            d_px = region.equivalent_diameter
            y, x = region.centroid
            rr, cc = ellipse_perimeter(int(y), int(x), int(d_px / 2), int(d_px / 2))
            rr = np.clip(rr, 0, original_image.shape[0] - 1)
            cc = np.clip(cc, 0, original_image.shape[1] - 1)
            circular_img[rr, cc] = (0, 255, 0)
            combined_img[rr, cc] = (0, 255, 0)

            # --- 3. Elliptical Equivalent Contour (in RED) ---
            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(elliptical_img, ellipse, (0, 0, 255), 1)
                    cv2.ellipse(combined_img, ellipse, (0, 0, 255), 1)

            # Add the diameter to the result list
            diameters_pixels.append(d_px)
            diameters_nm.append(d_nm)
            centroids.append(region.centroid)

    # Save the final image with all contour types
    # Save all images
    cv2.imwrite("outputs/figures/true_contours.png", true_contour_img)
    cv2.imwrite("outputs/figures/circular_equivalent.png", circular_img)
    cv2.imwrite("outputs/figures/elliptical_equivalent.png", elliptical_img)
    cv2.imwrite("outputs/figures/all_contour_types.png", combined_img)

    print("Saved all contour types:")
    print(" - outputs/figures/true_contours.png")
    print(" - outputs/figures/circular_equivalent.png")
    print(" - outputs/figures/elliptical_equivalent.png")
    print(" - outputs/figures/all_contour_types.png")

    # Convert the results to a DataFrame and save as CSV
    df = pd.DataFrame({"Diameter (nm)": diameters_nm})
    df.to_csv("outputs/results/nanoparticle_data.csv", index=False)

    # Return the list of diameters
    return diameters_nm, diameters_pixels, centroids

def export_to_latex(diameters, image_path):
    """
    Generates a LaTeX summary table of particle diameter statistics and writes it to a .tex file.

    Parameters:
    -----------
    diameters : list of float
        Measured particle diameters (in nanometers).
    image_path : str
        Path to the original image used for naming the output summary file.
    """
    
    import os
    import pandas as pd
    import numpy as np
    from scipy.stats import describe

    # Extract image name without extension to use in output filename
    base = os.path.splitext(os.path.basename(image_path))[0]

    # Compute summary statistics using scipy's describe
    stats = describe(diameters)

    # Construct a DataFrame with one row of summary statistics
    summary = pd.DataFrame([{
        "Image": base,
        "Count": stats.nobs,                                 # Number of particles
        "Mean": round(stats.mean, 2),                        # Mean diameter
        "Std Dev": round(np.sqrt(stats.variance), 2),        # Standard deviation
        "Min": round(stats.minmax[0], 2),                    # Minimum diameter
        "Max": round(stats.minmax[1], 2)                     # Maximum diameter
    }])

    # Define output .tex file path
    latex_path = f"outputs/results/{base}_summary.tex"

    # Export the summary table as LaTeX (tabular format)
    summary.to_latex(latex_path, index=False)
