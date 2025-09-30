"""
Pipeline Orchestrator for NanoPSD (Classical mode)
--------------------------------------------------
This module coordinates the overall analysis steps:
  1) Detect scale bar (pixels) and compute nm-per-pixel calibration.
  2) Preprocess image -> binary mask.
  3) Segment particles (Classical Otsu via OtsuSegmenter).
  4) Measure sizes (nm), generate plots, and export summary.

Notes
-----
* This version intentionally uses the Classical (Otsu) segmenter only.
  We wrapped it behind a small interface so AI can be dropped in later
  with zero changes to measurement/plot/export steps.

* Batch mode is supported: pass a directory instead of a file.
"""

# Import necessary modules
import logging
import os
from glob import glob
from typing import Tuple

# Import project-specific modules from their respective paths
from utils.scale_bar import (
    detect_scale_bar_length,
    detect_scale_bar,
    detect_scale_label,  # Hybrid detector + OCR
)
from scripts.preprocessing.clahe_filter import (
    preprocess_image,
)  # Enhances image using CLAHE and other filters
from scripts.segmentation.otsu_impl import (
    OtsuSegmenter,
)  # Classical Otsu-based segmenter for particle segmentation
from scripts.analysis.size_measurement import (
    measure_particles,
    export_to_latex,
)  # Measures particle sizes and exports
from scripts.visualization.plotting import (
    plot_results,
)  # Plots size distribution results

# Set up basic logging configuration
# This will show timestamps, logging levels, and messages in console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class NanoparticleAnalyzer:
    """Run the classical PSD pipeline on one image or a folder of images.

    Parameters
    ----------
    image_path : str
        Path to a single image (e.g., 'SEM.png') OR a directory to process in batch.
    scale_bar_nm : float
        Known physical length of the scale bar in nanometers (e.g., 100 for '100 nm').
    batch : bool, default False
        If True, process all matching images in `image_path` directory.
    extensions : Tuple[str, ...], default ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
        File extensions to consider in batch mode.
    min_size_px : int, default 3
        Minimum connected-component size (in pixels) to keep during segmentation.
    mode : str, default "classical"
        Segmentation mode selector. Only "classical" is implemented now.
    Later:
        "ai" for AI-only, "both" to run both, "compare" to run both and compare results.
    """

    def __init__(
        self,
        image_path: str,
        scale_bar_nm: float,
        batch: bool = False,
        extensions: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg"),
        min_size_px: int = 3,
        mode: str = "classical",
    ) -> None:
        self.image_path = image_path
        self.scale_bar_nm = float(scale_bar_nm)
        self.batch_mode = bool(batch)
        self.extensions = tuple(extensions)
        self.min_size_px = int(min_size_px)
        self.mode = mode  # Mode of analysis: 'classical', 'ai', 'both', 'compare'

        # Guard: only classical is implemented right now
        if self.mode != "classical":
            raise NotImplementedError(
                f"mode='{self.mode}' not implemented yet; use 'classical' for now"
            )

        # Classical segmenter wrapped behind an interface
        self.segmenter = OtsuSegmenter(min_size=self.min_size_px)

    # ----------------------------
    # Public API
    # ----------------------------
    def run(self) -> None:
        """Execute the pipeline on a single file or a folder (batch)."""
        if self.batch_mode and os.path.isdir(self.image_path):
            images = list(self._iter_images(self.image_path, self.extensions))
            if not images:
                logging.warning(f"No images found in folder: {self.image_path}")
                return
            logging.info(f"Batch mode: {len(images)} images found.")
            for img in images:
                self._process_one(img)
        else:
            self._process_one(self.image_path)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _iter_images(self, folder: str, exts: Tuple[str, ...]):
        """Yield images in `folder` that match any of the given extensions."""
        for ext in exts:
            for p in glob(os.path.join(folder, f"*{ext}")):
                yield p

    def _process_one(self, img_path: str) -> None:
        """Run the classical PSD pipeline on a single image path."""
        try:
            base = os.path.basename(img_path)
            logging.info(f"Processing: {base}")

            # --- Step 1: Detect scale bar (px) + bbox/mask; OCR label if scale not provided ---
            scale_bar_px, bar_bbox, bar_mask, _ = detect_scale_bar(
                img_path, save_debug=True, debug_dir="outputs/figures"
            )

            # Prefer CLI-provided scale if positive; otherwise attempt OCR
            eff_scale_nm = (
                float(self.scale_bar_nm)
                if getattr(self, "scale_bar_nm", None) is not None
                else None
            )
            if (eff_scale_nm is None) or (eff_scale_nm <= 0):
                try:
                    ocr_nm = detect_scale_label(
                        img_path, bar_bbox, save_debug=True, debug_dir="outputs/figures"
                    )
                    if ocr_nm:
                        eff_scale_nm = float(ocr_nm)
                except Exception:
                    pass

            # Final guard: must have a valid scale in nm at this point
            if (eff_scale_nm is None) or (eff_scale_nm <= 0):
                raise ValueError(
                    "No valid scale value found (neither CLI --scale nor OCR)."
                )

            nm_per_pixel = self._compute_nm_per_pixel(eff_scale_nm, scale_bar_px)
            logging.info(
                f"Calibration: {nm_per_pixel:.4f} nm/pixel (bar: {scale_bar_px} px, value: {eff_scale_nm} nm)"
            )

            # --- Step 2: Preprocess image -> binary mask ---
            binary, original = preprocess_image(img_path)

            # Mask out the scale bar region so it's never counted as a particle
            try:
                if "bar_mask" in locals() and bar_mask is not None:
                    binary = binary.copy()
                    binary[bar_mask > 0] = False
            except Exception:
                pass

            # --- Step 3: Segment particles (Classical Otsu via interface) ---
            labeled, regions = self.segmenter.segment(binary)
            logging.info(f"Segmented {len(regions)} regions (pre-filter).")

            # --- Step 4: Measure diameters in nm and overlay QA drawings (if any) ---
            diameters_nm, overlay_image, df = measure_particles(
                regions, labeled, original, nm_per_pixel, img_path
            )
            logging.info(f"Measured {len(diameters_nm)} particles (post-filter).")

            # --- Step 5: Visualize & Export ---
            plot_results(diameters_nm, img_path)
            export_to_latex(diameters_nm, img_path)

            logging.info(f"Completed: {base} | Count={len(diameters_nm)}")

        except Exception as e:
            logging.exception(f"Error while processing {img_path}: {e}")

    @staticmethod
    def _compute_nm_per_pixel(scale_bar_nm: float, scale_bar_px: float) -> float:
        """Return the conversion factor nm/pixel from a known scale bar length.

        Raises a ValueError if the detected pixel length is zero or negative.
        """
        if scale_bar_px <= 0:
            raise ValueError("Invalid scale bar pixel length (<= 0).")
        return scale_bar_nm / float(scale_bar_px)
