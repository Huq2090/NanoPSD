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

"""
Pipeline Orchestrator for NanoPSD (Classical Segmentation Mode)
===============================================================

This module coordinates the complete nanoparticle size distribution analysis:
  1. Scale bar detection (geometric + optional OCR)
  2. Image preprocessing (CLAHE enhancement, thresholding)
  3. Particle segmentation (Classical Otsu method)
  4. Size measurement (equivalent diameter in nanometers)
  5. Visualization & export (histograms, CSV, LaTeX summaries)

Architecture Notes:
-------------------
- Uses a modular design with clear separation of concerns
- Segmentation wrapped behind BaseSegmenter interface for easy AI integration later
- Supports both single-image and batch processing modes
- Comprehensive error handling with logging
- All measurements in nanometers (µm automatically converted)

Future Extensions:
------------------
- AI-based segmentation (mode="ai")
- Comparison mode (mode="compare") to evaluate classical vs AI
- Advanced morphology analysis (circularity, aspect ratio, etc.)
- Multi-scale analysis for images with multiple magnifications
"""

# Standard library imports
import logging
import os
from glob import glob
from typing import Tuple
import time

# Local project imports
from utils.scale_bar import (
    detect_scale_bar_length,  # Legacy wrapper (kept for compatibility)
    detect_scale_bar,  # Main geometric detector
    detect_scale_label,  # OCR text reader
)
from scripts.preprocessing.clahe_filter import preprocess_image
from scripts.segmentation.otsu_impl import OtsuSegmenter
from scripts.analysis.size_measurement import (
    measure_particles,
    export_to_latex,
    export_summary_csv,
)
from scripts.visualization.plotting import plot_results

# Configure logging format
# Shows timestamp, level (INFO/WARNING/ERROR), and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class NanoparticleAnalyzer:
    """
    Main pipeline orchestrator for nanoparticle size distribution analysis.

    This class encapsulates the complete analysis workflow from raw microscopy
    images to final statistical outputs. It handles both single images and
    batch processing of multiple images in a folder.

    Design Pattern:
    ---------------
    This is a "Facade" pattern - it provides a simple interface to a complex
    subsystem. Users only need to instantiate this class and call run().
    All the complexity of detection, segmentation, measurement is hidden.

    Attributes
    ----------
    image_path : str
        Path to single image or folder (depends on batch_mode)
    scale_bar_nm : float
        Physical length of scale bar in nanometers
    batch_mode : bool
        True for batch folder processing, False for single image
    extensions : tuple of str
        File extensions to process in batch mode
    min_size_px : int
        Minimum particle size in pixels (noise filter)
    mode : str
        Segmentation mode: "classical", "ai", "both", "compare"
        Currently only "classical" is implemented
    ocr_backend : str
        OCR engine selection: "easyocr-auto" or "easyocr-cpu"
    segmenter : BaseSegmenter
        Segmentation strategy object (currently OtsuSegmenter)

    Examples
    --------
    # Single image with manual scale
    >>> analyzer = NanoparticleAnalyzer(
    ...     image_path="sample.tif",
    ...     scale_bar_nm=200,
    ...     batch=False,
    ...     mode="classical"
    ... )
    >>> analyzer.run()

    # Batch processing with OCR
    >>> analyzer = NanoparticleAnalyzer(
    ...     image_path="./images/",
    ...     batch=True,
    ...     ocr_backend="easyocr-auto"
    ... )
    >>> analyzer.run()
    """

    def __init__(
        self,
        image_path: str,
        scale_bar_nm: float = None,  # Optional input flag
        batch: bool = False,
        extensions: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg"),
        min_size_px: int = 3,
        max_size_px: int = None,
        mode: str = "classical",
        ocr_backend: str = "easyocr-auto",
        verify_scale_bar: bool = False,
        nm_per_pixel: float = None,  # Adding capability to directly input nm_per_pixel
        save_preprocessing_steps: bool = False,
        save_segmentation_steps: bool = False,
        # Morphology classification thresholds
        spherical_ar_max=1.5,
        rodlike_ar_min=1.8,
        aggregate_c_max=0.60,
        spherical_c_min=0.75,
        rodlike_s_min=0.80,
        aggregate_s_max=0.85,
        spherical_s_min=0.90,
    ) -> None:
        """
        Initialize the analyzer with user parameters.

        Parameters
        ----------
        image_path : str
            Path to input:
            - Single mode: path to one image file (e.g., "sample.tif")
            - Batch mode: path to folder containing images (e.g., "./images/")

        scale_bar_nm : float, optional
            Physical length of the scale bar in nanometers
            - Positive value (e.g., 200.0): use this as known scale
            - 0: invalid, will cause error
            - None: not using scale bar (must provide nm_per_pixel)

        nm_per_pixel : float, optional
            - Direct calibration factor for images without scale bars.
            - When provided, scale bar detection is skipped entirely.

            NOTE: Provide EITHER scale_bar_nm OR nm_per_pixel (not both)

        batch : bool, optional
            Processing mode selection (default: False)
            - False: process single image specified by image_path
            - True: process all matching images in folder specified by image_path

        extensions : tuple of str, optional
            File extensions to process in batch mode (default: common formats)
            Only used when batch=True
            Examples: (".png", ".tif"), (".jpg", ".jpeg", ".tiff")

        min_size_px : int, optional
            Minimum particle size in pixels (default: 3)
            Particles smaller than this are filtered out as noise
            Typical values:
            - 3-5: high-resolution, clean images
            - 5-10: noisy or lower-resolution images
            - 10+: when focusing only on larger particles

        mode : str, optional
            Segmentation algorithm selection (default: "classical")
            Options:
            - "classical": Otsu thresholding (fast, works for most images)
            - "ai": Deep learning segmentation (not implemented yet)
            - "both": Run both and use AI results (not implemented yet)
            - "compare": Run both and compare (not implemented yet)

        ocr_backend : str, optional
            OCR backend for scale bar text reading (default: "easyocr-auto")
            Only used when --ocr-backend flag is provided
            Options:
            - "easyocr-auto": Auto-detect GPU, fallback to CPU (recommended)
            - "easyocr-cpu": Force CPU-only processing (useful if GPU causes issues)

        Raises
        ------
        NotImplementedError
            If mode is not "classical" (other modes planned but not ready)
        """
        # Store all parameters as instance attributes
        self.image_path = image_path
        self.scale_bar_nm = float(scale_bar_nm) if scale_bar_nm is not None else None
        self.batch_mode = bool(batch)
        self.extensions = tuple(extensions)
        self.min_size_px = int(min_size_px)
        self.max_size_px = int(max_size_px) if max_size_px is not None else None
        self.mode = mode
        self.ocr_backend = ocr_backend  # Store OCR backend choice
        self.verify_scale_bar = verify_scale_bar  # Store scale bar verification flag
        self.nm_per_pixel_manual = nm_per_pixel  # Direct nm/pixel input
        self.save_preprocessing_steps = save_preprocessing_steps
        self.save_segmentation_steps = save_segmentation_steps

        # Store results for batch aggregation
        self.batch_results = []  # Will hold DataFrames from each image
        self.individual_times = []  # Track processing times for batch mode

        # Store morphology thresholds
        self.spherical_ar_max = spherical_ar_max
        self.rodlike_ar_min = rodlike_ar_min
        self.aggregate_c_max = aggregate_c_max
        self.spherical_c_min = spherical_c_min
        self.rodlike_s_min = rodlike_s_min
        self.aggregate_s_max = aggregate_s_max
        self.spherical_s_min = spherical_s_min

        # Validate: exactly one calibration method
        has_scale_bar = self.scale_bar_nm is not None
        has_ocr = self.ocr_backend is not None
        has_nm_per_px = self.nm_per_pixel_manual is not None

        # Count how many methods provided
        methods_count = sum([has_scale_bar, has_ocr, has_nm_per_px])

        if methods_count == 0:
            raise ValueError(
                "Must provide ONE calibration method:\n"
                "  --scale-bar-nm VALUE     (manual scale value)\n"
                "  --ocr-backend BACKEND    (automatic OCR detection)\n"
                "  --nm-per-pixel VALUE     (no scale bar)"
            )

        if methods_count > 1:
            methods_used = []
            if has_scale_bar:
                methods_used.append("scale_bar_nm")
            if has_ocr:
                methods_used.append("ocr_backend")
            if has_nm_per_px:
                methods_used.append("nm_per_pixel")

            raise ValueError(
                f"Cannot use multiple calibration methods: {', '.join(methods_used)}\n"
                f"Choose only ONE method."
            )

        # Guard against unimplemented modes
        if self.mode != "classical":
            raise NotImplementedError(
                f"mode='{self.mode}' not implemented yet; use 'classical' for now"
            )

        # Instantiate the segmentation strategy
        # Using Strategy pattern: segmenter implements BaseSegmenter interface
        # This makes it easy to swap in AI segmentation later without changing this code
        self.segmenter = OtsuSegmenter(
            min_size=self.min_size_px,
            max_size=self.max_size_px,
            save_steps=self.save_segmentation_steps,
            output_dir="outputs/segmentation_steps",
            image_name="placeholder",  # Will be updated per image
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def run(self) -> None:
        """
        Execute the analysis pipeline on single image or batch folder.

        This is the main entry point after initialization. It:
        1. Determines if batch or single mode
        2. Finds all images to process
        3. Calls _process_one() for each image
        4. Logs progress and any errors

        Batch Mode Behavior:
        --------------------
        - Finds all files matching self.extensions in self.image_path folder
        - Processes each image independently
        - Errors in one image don't stop processing of others
        - Progress logged for each image

        Single Mode Behavior:
        ---------------------
        - Processes the single image at self.image_path
        - Any error stops execution (appropriate for single file)

        Returns
        -------
        None
            Results written to outputs/results/ and outputs/figures/

        Side Effects
        ------------
        - Writes CSV files to outputs/results/
        - Writes PNG/TIF images to outputs/figures/
        - Writes LaTeX summaries to outputs/results/
        - Logs progress messages to console
        """
        if self.batch_mode and os.path.isdir(self.image_path):
            # Batch mode: find all matching images in folder
            images = list(self._iter_images(self.image_path, self.extensions))

            if not images:
                logging.warning(f"No images found in folder: {self.image_path}")
                return

            logging.info(f"Batch mode: {len(images)} images found.")

            batch_start_time = time.time()  # Start batch timing
            self.individual_times = []  # Reset for this batch

            # Process each image (errors caught per-image, don't stop batch)
            for img in images:
                self._process_one(img)

            # Generate batch aggregation outputs
            if self.batch_results:
                self._generate_batch_report(batch_start_time)
            else:
                logging.warning("No results to aggregate in batch mode")
        else:
            # Single mode: process one image
            self._process_one(self.image_path)

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _iter_images(self, folder: str, exts: Tuple[str, ...]):
        """
        Generator that yields image paths in folder matching extensions.

        This uses glob to find files, which handles wildcards naturally.
        Using a generator (yield) instead of building a list is more
        memory-efficient for large batches.

        Parameters
        ----------
        folder : str
            Path to folder to search
        exts : tuple of str
            File extensions to match (e.g., (".png", ".tif"))

        Yields
        ------
        str
            Full path to each matching image file

        Examples
        --------
        >>> for img in self._iter_images("./data/", (".tif", ".png")):
        ...     print(img)
        ./data/sample1.tif
        ./data/sample2.png
        """
        for ext in exts:
            # Glob pattern: folder/*{ext}
            # Example: ./images/*.tif
            pattern = os.path.join(folder, f"*{ext}")
            for p in glob(pattern):
                yield p

    def _process_one(self, img_path: str) -> None:
        """
        Run the complete analysis pipeline on a single image.

        This is where the actual work happens. It coordinates all the modules:
        - utils.scale_bar for scale detection
        - scripts.preprocessing for image enhancement
        - scripts.segmentation for particle identification
        - scripts.analysis for size measurement
        - scripts.visualization for plotting

        Pipeline Steps:
        ---------------
        1. Detect scale bar (geometric detection + bounding box + mask)
        2. Determine physical scale:
        - If user provided --scale-bar-nm N: use N
        - If user do not provided --scale-bar-nm: look for --ocr-backend options
        - OCR uses the specified backend (easyocr-auto/easyocr-cpu)
        3. Compute nm-per-pixel calibration factor
        4. Preprocess image (CLAHE + threshold → binary mask)
        5. Mask out scale bar region (prevent it being counted as particle)
        6. Segment particles (find connected components)
        7. Measure particle diameters in nanometers
        8. Generate plots and export results

        Parameters
        ----------
        img_path : str
            Path to image file to process

        Returns
        -------
        None
            Results written to outputs/ directory

        Raises
        ------
        ValueError
            If scale bar detection fails or no valid scale value found
        Exception
            Any other errors during processing (caught and logged)

        Side Effects
        ------------
        - Writes files to outputs/results/ and outputs/figures/
        - Logs progress messages
        - Prints OCR success/failure messages
        """
        try:
            base = os.path.basename(img_path)
            logging.info(f"Processing: {base}")
            start_time = time.time()  # Start timing for performance measurement
            # ----------------------------------------------------------------------------------------

            # -----------------------------------------------------------------
            # Step 1: Determine Calibration Mode (3 options)
            # -----------------------------------------------------------------

            if self.nm_per_pixel_manual is not None:
                # MODE A: Manual calibration (no scale bar)
                logging.info("⚙️  Manual calibration mode (no scale bar)")
                nm_per_pixel = self.nm_per_pixel_manual
                bar_mask = None
                logging.info(f"Calibration: {nm_per_pixel:.4f} nm/pixel (manual)")

            elif self.ocr_backend is not None:
                # MODE B: OCR auto-detection
                logging.info("🔍 OCR auto-detection mode")

                # Step 1: Detect scale bar (geometric detection)
                scale_bar_px, bar_bbox, bar_mask, _ = detect_scale_bar(
                    img_path, save_debug=True, debug_dir="outputs/figures"
                )

                # Step 2: Use OCR to read text
                text_bbox = None
                try:
                    ocr_result = detect_scale_label(
                        img_path,
                        bar_bbox,
                        save_debug=True,
                        debug_dir="outputs/figures",
                        ocr_backend=self.ocr_backend,
                    )

                    if ocr_result:
                        if isinstance(ocr_result, tuple):
                            eff_scale_nm, text_bbox = ocr_result
                        else:
                            eff_scale_nm = ocr_result

                        if text_bbox:
                            logging.info(f"OCR detected text at: {text_bbox}")
                    else:
                        raise ValueError("OCR failed to detect scale bar text")

                except Exception as e:
                    raise ValueError(
                        f"OCR error: {e}\nTry using --scale-bar-nm with manual value instead."
                    )

                # Verification if requested
                if self.verify_scale_bar:
                    if not self._show_verification(
                        img_path, bar_bbox, scale_bar_px, eff_scale_nm
                    ):
                        logging.warning(f"User rejected: {base}")
                        return

                # Compute calibration factor
                nm_per_pixel = self._compute_nm_per_pixel(eff_scale_nm, scale_bar_px)
                logging.info(
                    f"Calibration: {nm_per_pixel:.4f} nm/pixel "
                    f"(bar: {scale_bar_px}px, OCR detected: {eff_scale_nm}nm)"
                )

            else:
                # MODE C: Manual scale bar value
                logging.info("📏 Manual scale bar mode")

                # Step 1: Detect scale bar (geometric detection)
                scale_bar_px, bar_bbox, bar_mask, _ = detect_scale_bar(
                    img_path, save_debug=True, debug_dir="outputs/figures"
                )

                # Step 2: Use user-provided value
                eff_scale_nm = float(self.scale_bar_nm)
                text_bbox = None

                # Verification if requested
                if self.verify_scale_bar:
                    if not self._show_verification(
                        img_path, bar_bbox, scale_bar_px, eff_scale_nm
                    ):
                        logging.warning(f"User rejected: {base}")
                        return

                # Step 3: Compute calibration factor
                nm_per_pixel = self._compute_nm_per_pixel(eff_scale_nm, scale_bar_px)
                logging.info(
                    f"Calibration: {nm_per_pixel:.4f} nm/pixel "
                    f"(bar: {scale_bar_px}px, value: {eff_scale_nm}nm)"
                )

            # -----------------------------------------------------------------
            # Step 4: Preprocess image to binary mask
            # -----------------------------------------------------------------
            # Returns:
            # - binary: boolean array (True = particle, False = background)
            # - original: grayscale image (for overlay visualization later)
            # binary, original = preprocess_image(img_path)
            binary, original = preprocess_image(
                img_path,
                save_steps=(
                    self.save_preprocessing_steps
                    if hasattr(self, "save_preprocessing_steps")
                    else False
                ),
            )

            # -----------------------------------------------------------------
            # Step 5: Mask out scale bar region
            # -----------------------------------------------------------------
            # The scale bar would otherwise be detected as a huge "particle"
            # We set all pixels in the bar region to False (background)
            if bar_mask is not None:
                logging.info("Excluding scale bar region from particle detection...")
                binary = binary.copy()
                binary[bar_mask > 0] = False

                # Tier 1: Use OCR-detected text location (most precise)
                if "text_bbox" in locals() and text_bbox is not None:
                    tx, ty, tw, th = text_bbox
                    pad = 15
                    ty1 = max(0, ty - pad)
                    ty2 = min(binary.shape[0], ty + th + pad)
                    tx1 = max(0, tx - pad)
                    tx2 = min(binary.shape[1], tx + tw + pad)

                    binary[ty1:ty2, tx1:tx2] = False
                    logging.info(f"Excluded OCR text at ({tx},{ty},{tw},{th})")

                # Tier 2: Geometric exclusion around bar (for manual scale mode)
                elif "bar_bbox" in locals():
                    logging.info("Using geometric text exclusion around scale bar")
                    bx, by, bw, bh = bar_bbox

                    pad_x = 200
                    pad_y = 50

                    tx1 = max(0, bx - pad_x)
                    ty1 = max(0, by - pad_y)
                    tx2 = min(binary.shape[1], bx + bw + pad_x)
                    ty2 = min(binary.shape[0], by + bh + pad_y)

                    binary[ty1:ty2, tx1:tx2] = False
                    logging.info(f"Excluded area: ({tx1},{ty1}) to ({tx2},{ty2})")

                # Tier 3: Fallback if no bar detected at all
                else:
                    logging.warning("No scale bar detected - excluding bottom 15%")
                    h, w = binary.shape
                    binary[int(h * 0.85) :, :] = False
            else:
                # No scale bar - Skip all Masking
                logging.info("✓ No scale bar masking needed (manual calibration)")

            # -----------------------------------------------------------------
            # Step 6: Segment particles
            # -----------------------------------------------------------------
            # The segmenter (OtsuSegmenter) returns:
            # - labeled: integer array where each particle has unique ID
            # - regions: list of regionprops (area, centroid, etc.)
            # Update image name for save_steps
            if self.save_segmentation_steps:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                self.segmenter.image_name = base_name

            labeled, regions = self.segmenter.segment(binary)
            logging.info(f"Segmented {len(regions)} regions after exclusion.")

            # -----------------------------------------------------------------
            # Step 7: Measure particle diameters in nanometers
            # -----------------------------------------------------------------
            # Also generates overlay visualizations showing contours
            # Returns:
            # - diameters_nm: list of floats (diameter in nm for each particle)
            # - overlay_image: annotated image with contours
            # - df: pandas DataFrame with measurements (unused here)
            diameters_nm, overlay_image, df = measure_particles(
                regions,
                labeled,
                original,
                nm_per_pixel,
                img_path,
                min_size_px=self.min_size_px,
                max_size_px=self.max_size_px,
                # Pass morphology thresholds
                spherical_ar_max=self.spherical_ar_max,
                spherical_c_min=self.spherical_c_min,
                spherical_s_min=self.spherical_s_min,
                aggregate_s_max=self.aggregate_s_max,
                aggregate_c_max=self.aggregate_c_max,
                rodlike_ar_min=self.rodlike_ar_min,
                rodlike_s_min=self.rodlike_s_min,
            )
            logging.info(f"Measured {len(diameters_nm)} particles (post-filter).")

            # Calculate and display processing time
            processing_time = time.time() - start_time
            logging.info(f"Image processing time: {processing_time:.2f} seconds")

            # -----------------------------------------------------------------
            # Step 8: Visualize and export results
            # -----------------------------------------------------------------
            # plot_results: generates histograms (overall + morphology)
            plot_results(diameters_nm, img_path, df=df)

            # export_to_latex: generates statistical summary table
            export_to_latex(diameters_nm, img_path)

            # Export summary CSV (SINGLE MODE ONLY)
            if not self.batch_mode:
                export_summary_csv(diameters_nm, df, img_path)

            # Print morphology summary
            if len(df) > 0 and "Morphology" in df.columns:
                print("\n" + "=" * 60)
                print("MORPHOLOGY SUMMARY")
                print("=" * 60)
                morphology_counts = df["Morphology"].value_counts()
                for morph in ["spherical", "rod-like", "aggregate"]:
                    count = morphology_counts.get(morph, 0)
                    percentage = (count / len(df)) * 100
                    avg_diam = (
                        df[df["Morphology"] == morph]["Diameter (nm)"].mean()
                        if count > 0
                        else 0
                    )
                    print(
                        f"{morph.capitalize():12s}: {count:4d} ({percentage:5.1f}%)  "
                        f"Avg: {avg_diam:6.2f} nm"
                    )
                print("=" * 60)

            logging.info(f"Completed: {base} | Count={len(diameters_nm)}")

            # Track processing time for batch statistics
            if self.batch_mode:
                self.individual_times.append(processing_time)

            # Store results for batch aggregation
            if self.batch_mode and len(df) > 0:
                df_copy = df.copy()
                df_copy["Image"] = base  # Add source image column
                self.batch_results.append(df_copy)

        except Exception as e:
            # Catch and log any errors during processing
            # This prevents one bad image from crashing batch mode
            logging.exception(f"Error while processing {img_path}: {e}")

    def _generate_batch_report(self, batch_start_time) -> None:
        """
        Generate aggregate outputs for batch processing.

        Creates:
        - Combined CSV with all particles
        - Summary statistics table
        - Comparison visualizations
        """
        import pandas as pd
        from scripts.visualization.plotting import plot_batch_comparison

        logging.info(f"\n{'='*60}")
        logging.info("GENERATING BATCH REPORT")
        logging.info(f"{'='*60}")

        # Combine all dataframes
        df_all = pd.concat(self.batch_results, ignore_index=True)

        # Save combined CSV
        combined_csv = "outputs/results/batch_all_particles.csv"
        df_all.to_csv(combined_csv, index=False)
        logging.info(f"Saved combined CSV: {combined_csv}")

        # Generate summary statistics per image
        summary_data = []
        for img_name in df_all["Image"].unique():
            img_df = df_all[df_all["Image"] == img_name]

            summary_data.append(
                {
                    "Image": img_name,
                    "Total_Particles": len(img_df),
                    "Mean_Diameter_nm": img_df["Diameter (nm)"].mean(),
                    "Std_Diameter_nm": img_df["Diameter (nm)"].std(),
                    "Median_Diameter_nm": img_df["Diameter (nm)"].median(),
                    "Min_Diameter_nm": img_df["Diameter (nm)"].min(),
                    "Max_Diameter_nm": img_df["Diameter (nm)"].max(),
                    "Spherical_Count": len(img_df[img_df["Morphology"] == "spherical"]),
                    "RodLike_Count": len(img_df[img_df["Morphology"] == "rod-like"]),
                    "Aggregate_Count": len(img_df[img_df["Morphology"] == "aggregate"]),
                }
            )

        df_summary = pd.DataFrame(summary_data)
        summary_csv = "outputs/results/batch_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        logging.info(f"Saved summary statistics: {summary_csv}")

        # Print summary table
        print(f"\n{'='*80}")
        print("BATCH SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(df_summary.to_string(index=False))
        print(f"{'='*80}")
        print(f"Total Images Processed: {len(df_summary)}")
        print(f"Total Particles Detected: {len(df_all)}")
        print(f"Overall Mean Diameter: {df_all['Diameter (nm)'].mean():.2f} nm")
        print(f"Overall Std Deviation: {df_all['Diameter (nm)'].std():.2f} nm")
        print(f"{'='*80}\n")

        # Generate comparison plots
        plot_batch_comparison(df_all, df_summary)

        # Display batch timing statistics
        total_batch_time = time.time() - batch_start_time
        avg_time = (
            sum(self.individual_times) / len(self.individual_times)
            if self.individual_times
            else 0
        )

        logging.info("\n" + "=" * 60)
        logging.info("BATCH PROCESSING TIME SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total images processed: {len(self.individual_times)}")
        logging.info(f"Total processing time: {total_batch_time:.2f} seconds")
        logging.info(f"Average time per image: {avg_time:.2f} seconds")
        logging.info("=" * 60 + "\n")

        logging.info("Batch report generation complete")

    def _show_verification(self, img_path, bar_bbox, scale_bar_px, ocr_value):
        """Show scale bar crop and wait for Y/N."""
        import cv2

        img = cv2.imread(img_path)
        if img is None:
            return True

        x, y, w, h = bar_bbox

        # Crop 150px around bar
        y1 = max(0, y - 200)
        y2 = min(img.shape[0], y + h + 200)
        x1 = max(0, x - 200)
        x2 = min(img.shape[1], x + w + 200)

        crop = img[y1:y2, x1:x2].copy()

        # Draw green box
        cv2.rectangle(crop, (x - x1, y - y1), (x - x1 + w, y - y1 + h), (0, 0, 200), 3)

        # Add text
        text = f"{scale_bar_px}px"
        if ocr_value:
            text += f" = {ocr_value}nm"
        cv2.putText(crop, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.putText(
            crop,
            "Y=Accept N=Reject",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Verify Scale Bar", crop)
        print("\n" + "=" * 60)
        print("CLICK IMAGE WINDOW then press Y or N")
        print("=" * 60 + "\n")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("y") or key == ord("Y"):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return True
            elif key == ord("n") or key == ord("N"):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                return False

    @staticmethod
    def _compute_nm_per_pixel(scale_bar_nm: float, scale_bar_px: float) -> float:
        """
        Compute the calibration factor (nanometers per pixel).

        This is simple division, but we validate inputs to catch errors early
        and provide helpful error messages.

        Formula:
        --------
        nm_per_pixel = scale_bar_nm / scale_bar_px

        Example:
        --------
        If scale bar is 200 nm and measures 100 pixels:
        nm_per_pixel = 200 / 100 = 2.0

        So each pixel represents 2 nanometers in physical space.

        Parameters
        ----------
        scale_bar_nm : float
            Physical length of scale bar in nanometers
        scale_bar_px : float
            Measured length of scale bar in pixels

        Returns
        -------
        float
            Conversion factor (nm / pixel)

        Raises
        ------
        ValueError
            If scale_bar_px is zero or negative (invalid measurement)

        Examples
        --------
        >>> _compute_nm_per_pixel(200, 100)
        2.0

        >>> _compute_nm_per_pixel(50, 250)
        0.2

        >>> _compute_nm_per_pixel(100, 0)
        ValueError: Invalid scale bar pixel length (<= 0).
        """
        if scale_bar_px <= 0:
            raise ValueError("Invalid scale bar pixel length (<= 0).")

        return scale_bar_nm / float(scale_bar_px)
