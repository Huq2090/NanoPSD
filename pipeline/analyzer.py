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

# Local project imports
from utils.scale_bar import (
    detect_scale_bar_length,  # Legacy wrapper (kept for compatibility)
    detect_scale_bar,  # Main geometric detector
    detect_scale_label,  # OCR text reader
)
from scripts.preprocessing.clahe_filter import preprocess_image
from scripts.segmentation.otsu_impl import OtsuSegmenter
from scripts.analysis.size_measurement import measure_particles, export_to_latex
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
        Use -1 to enable OCR auto-detection
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
        OCR engine selection: "auto", "easyocr", or "tesseract"
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
    ...     scale_bar_nm=-1,  # Use OCR
    ...     batch=True,
    ...     ocr_backend="easyocr"
    ... )
    >>> analyzer.run()
    """

    def __init__(
        self,
        image_path: str,
        scale_bar_nm: float,
        batch: bool = False,
        extensions: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg"),
        min_size_px: int = 3,
        mode: str = "classical",
        ocr_backend: str = "auto",
        verify_scale_bar: bool = False,
    ) -> None:
        """
        Initialize the analyzer with user parameters.

        Parameters
        ----------
        image_path : str
            Path to input:
            - Single mode: path to one image file (e.g., "sample.tif")
            - Batch mode: path to folder containing images (e.g., "./images/")

        scale_bar_nm : float
            Physical length of the scale bar in nanometers
            - Positive value (e.g., 200.0): use this as known scale
            - -1: attempt automatic OCR detection of scale bar text
            - 0: invalid, will cause error

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
            OCR engine for scale bar text reading (default: "auto")
            Only used when scale_bar_nm=-1
            Options:
            - "auto": try EasyOCR first, fall back to Tesseract
            - "easyocr": use only EasyOCR (most accurate)
            - "tesseract": use only Tesseract (faster but less accurate)

        Raises
        ------
        NotImplementedError
            If mode is not "classical" (other modes planned but not ready)
        """
        # Store all parameters as instance attributes
        self.image_path = image_path
        self.scale_bar_nm = float(scale_bar_nm)
        self.batch_mode = bool(batch)
        self.extensions = tuple(extensions)
        self.min_size_px = int(min_size_px)
        self.mode = mode
        self.ocr_backend = ocr_backend  # Store OCR backend choice
        self.verify_scale_bar = verify_scale_bar  # Store scale bar verification flag

        # Guard against unimplemented modes
        if self.mode != "classical":
            raise NotImplementedError(
                f"mode='{self.mode}' not implemented yet; use 'classical' for now"
            )

        # Instantiate the segmentation strategy
        # Using Strategy pattern: segmenter implements BaseSegmenter interface
        # This makes it easy to swap in AI segmentation later without changing this code
        self.segmenter = OtsuSegmenter(min_size=self.min_size_px)

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

            # Process each image (errors caught per-image, don't stop batch)
            for img in images:
                self._process_one(img)
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
           - If user provided --scale-bar-nm -1: attempt OCR
           - OCR uses the specified backend (auto/easyocr/tesseract)
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

            # -----------------------------------------------------------------
            # Step 1: Detect scale bar (geometric detection)
            # -----------------------------------------------------------------
            # Returns:
            # - scale_bar_px: length of bar in pixels
            # - bar_bbox: (x, y, w, h) bounding box coordinates
            # - bar_mask: binary mask (255 inside bar region, 0 elsewhere)
            # - _: visualization (not used here, saved by detect_scale_bar)
            scale_bar_px, bar_bbox, bar_mask, _ = detect_scale_bar(
                img_path, save_debug=True, debug_dir="outputs/figures"
            )

            # -----------------------------------------------------------------
            # Step 2: Determine effective scale value (manual or OCR)
            # -----------------------------------------------------------------
            # Start with user-provided value
            eff_scale_nm = float(self.scale_bar_nm) if self.scale_bar_nm > 0 else None

            # Initialize text_bbox (will be set by OCR if used)
            text_bbox = None

            # If scale not provided or is -1, try OCR
            if eff_scale_nm is None or eff_scale_nm <= 0:
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
                            ocr_nm, text_bbox = ocr_result
                        else:
                            ocr_nm = ocr_result

                        if ocr_nm:
                            eff_scale_nm = float(ocr_nm)
                            if text_bbox:
                                logging.info(f"OCR found text at: {text_bbox}")
                except Exception as e:
                    logging.warning(f"OCR error: {e}")

            # Verification if requested
            if self.verify_scale_bar:
                if not self._show_verification(
                    img_path, bar_bbox, scale_bar_px, eff_scale_nm
                ):
                    logging.warning(f"User rejected: {base}")
                    return

            # Final validation: must have a valid scale at this point
            if (eff_scale_nm is None) or (eff_scale_nm <= 0):
                raise ValueError(
                    "No valid scale value found (neither CLI --scale-bar-nm nor OCR)."
                )

            # -----------------------------------------------------------------
            # Step 3: Compute calibration factor (nm per pixel)
            # -----------------------------------------------------------------
            nm_per_pixel = self._compute_nm_per_pixel(eff_scale_nm, scale_bar_px)
            logging.info(
                f"Calibration: {nm_per_pixel:.4f} nm/pixel "
                f"(bar: {scale_bar_px} px, value: {eff_scale_nm} nm)"
            )

            # -----------------------------------------------------------------
            # Step 4: Preprocess image to binary mask
            # -----------------------------------------------------------------
            # Returns:
            # - binary: boolean array (True = particle, False = background)
            # - original: grayscale image (for overlay visualization later)
            binary, original = preprocess_image(img_path)

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
                if text_bbox is not None:
                    tx, ty, tw, th = text_bbox
                    pad = 15
                    ty1 = max(0, ty - pad)
                    ty2 = min(binary.shape[0], ty + th + pad)
                    tx1 = max(0, tx - pad)
                    tx2 = min(binary.shape[1], tx + tw + pad)

                    binary[ty1:ty2, tx1:tx2] = False
                    logging.info(f"Excluded OCR text at ({tx},{ty},{tw},{th})")

                # Tier 2: Geometric exclusion around bar (for manual scale mode)
                else:
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

            # -----------------------------------------------------------------
            # Step 6: Segment particles
            # -----------------------------------------------------------------
            # The segmenter (OtsuSegmenter) returns:
            # - labeled: integer array where each particle has unique ID
            # - regions: list of regionprops (area, centroid, etc.)
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
                regions, labeled, original, nm_per_pixel, img_path
            )
            logging.info(f"Measured {len(diameters_nm)} particles (post-filter).")

            # -----------------------------------------------------------------
            # Step 8: Visualize and export results
            # -----------------------------------------------------------------
            # plot_results: generates histograms (overall + morphology)
            plot_results(diameters_nm, img_path, df=df)

            # export_to_latex: generates statistical summary table
            export_to_latex(diameters_nm, img_path)

            # === NEW: Print morphology summary ===
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

        except Exception as e:
            # Catch and log any errors during processing
            # This prevents one bad image from crashing batch mode
            logging.exception(f"Error while processing {img_path}: {e}")

    def _show_verification(self, img_path, bar_bbox, scale_bar_px, ocr_value):
        """Show scale bar crop and wait for Y/N."""
        import cv2

        img = cv2.imread(img_path)
        if img is None:
            return True

        x, y, w, h = bar_bbox

        # Crop 150px around bar
        y1 = max(0, y - 150)
        y2 = min(img.shape[0], y + h + 150)
        x1 = max(0, x - 150)
        x2 = min(img.shape[1], x + w + 150)

        crop = img[y1:y2, x1:x2].copy()

        # Draw green box
        cv2.rectangle(crop, (x - x1, y - y1), (x - x1 + w, y - y1 + h), (0, 255, 0), 3)

        # Add text
        text = f"{scale_bar_px}px"
        if ocr_value:
            text += f" = {ocr_value}nm"
        cv2.putText(crop, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(
            crop,
            "Y=Accept N=Reject",
            (10, crop.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
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
