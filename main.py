#!/usr/bin/env python3
"""
NanoPSD - Main Entry Point
===========================

This is the primary entry point for the NanoPSD (Nanoparticle Size Distribution)
analysis pipeline. It orchestrates the complete workflow from image input to
final statistical output.

Workflow Overview:
------------------
1. Parse command-line arguments (mode, input path, scale, etc.)
2. Create output directories if they don't exist
3. Instantiate the NanoparticleAnalyzer with user parameters
4. Run the analysis pipeline (single image or batch)
5. Generate results (CSV tables, histograms, LaTeX summaries)

Design Philosophy:
------------------
This file is intentionally minimal - it only handles:
- Argument parsing delegation (via scripts.cli)
- Output directory setup
- Analyzer instantiation and execution

All the heavy lifting (detection, segmentation, measurement) happens in the
pipeline.analyzer module, keeping this entry point clean and maintainable.

Usage Examples:
---------------
# Single image with manual scale
python3 main.py --mode single --input sample.tif --scale 200 --algo classical --min-size 3

# Single image with automatic OCR detection (EasyOCR)
python3 main.py --mode single --input sample.tif --scale -1 --algo classical --min-size 3 --ocr-backend easyocr

# Batch processing with auto OCR
python3 main.py --mode batch --input ./images/ --scale -1 --algo classical --min-size 5 --ocr-backend auto

For detailed help:
python3 main.py --help
"""

# Standard library imports
import os
import sys

# Local imports
from scripts.cli import parse_args
from pipeline.analyzer import NanoparticleAnalyzer


def main() -> None:
    """
    Main execution function for the NanoPSD pipeline.

    This function coordinates the entire analysis workflow:
    1. Parse CLI arguments
    2. Validate inputs (done by argparse)
    3. Setup output directories
    4. Create analyzer instance with user parameters
    5. Execute the analysis pipeline

    Returns
    -------
    None
        Results are written to files in outputs/ directory

    Raises
    ------
    SystemExit
        If invalid arguments provided (handled by argparse)
    Exception
        Any errors during analysis are caught and logged by the analyzer

    Output Structure:
    -----------------
    outputs/
    ├── results/
    │   ├── nanoparticle_data.csv        # Particle diameters
    │   └── {image_name}_summary.tex     # Statistical summary (LaTeX)
    └── figures/
        ├── {image_name}_diameter_histogram.png
        ├── {image_name}_true_contours.{ext}
        ├── {image_name}_circular_equivalent.{ext}
        ├── {image_name}_elliptical_equivalent.{ext}
        ├── {image_name}_all_contour_types.{ext}
        ├── scale_bar_final.png              # Debug: detected scale bar
        └── scale_candidates.png             # Debug: scale bar ROI
    """

    # -------------------------------------------------------------------------
    # Step 1: Parse command-line arguments
    # -------------------------------------------------------------------------
    # This delegates to scripts/cli.py which defines all CLI options
    # If arguments are invalid, argparse will print help and exit automatically
    args = parse_args()

    # -------------------------------------------------------------------------
    # Step 2: Setup output directory structure
    # -------------------------------------------------------------------------
    # Create output directories if they don't exist
    # exist_ok=True: don't raise error if directory already exists
    os.makedirs("outputs/results", exist_ok=True)  # For CSV and LaTeX files
    os.makedirs("outputs/figures", exist_ok=True)  # For plots and visualizations

    # -------------------------------------------------------------------------
    # Step 3: Instantiate the analyzer with user parameters
    # -------------------------------------------------------------------------
    # The NanoparticleAnalyzer is the core orchestrator that:
    # - Detects scale bars (geometry + optional OCR)
    # - Preprocesses images (CLAHE enhancement)
    # - Segments particles (currently Otsu, AI planned)
    # - Measures particle sizes (equivalent diameter)
    # - Generates plots and exports results

    analyzer = NanoparticleAnalyzer(
        # Input configuration
        image_path=args.input,  # File path or folder path
        batch=(args.mode == "batch"),  # Single vs batch mode
        # Scale calibration
        scale_bar_nm=args.scale_bar_nm,  # Physical scale in nm (-1 = use OCR)
        # Segmentation parameters
        mode=args.algo,  # "classical" (only one implemented now)
        min_size_px=args.min_size,  # Minimum particle size filter (pixels)
        # OCR configuration
        ocr_backend=args.ocr_backend,  # "auto", "easyocr", or "tesseract"
        verify_scale_bar=args.verify_scale_bar,  # Enable manual verification of scale detection
    )

    # -------------------------------------------------------------------------
    # Step 4: Run the analysis pipeline
    # -------------------------------------------------------------------------
    # This executes the full workflow:
    # - Single mode: process one image
    # - Batch mode: iterate over all images in folder
    #
    # All errors are caught and logged by the analyzer, so the program
    # won't crash on a single bad image in batch mode
    analyzer.run()

    # -------------------------------------------------------------------------
    # Step 5: Done!
    # -------------------------------------------------------------------------
    # Results have been written to outputs/ directory
    # The analyzer logs progress messages, so the user can see what happened


if __name__ == "__main__":
    """
    Standard Python idiom: only run main() if this file is executed directly.

    This allows the module to be imported for testing without running main().
    Example:
        python3 main.py            # Runs main()
        import main                # Does NOT run main()
        from main import main      # Does NOT run main()
    """
    main()
