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

#!/usr/bin/env python3
"""
NanoPSD - Main Entry Point
===========================

This is the primary entry point for the NanoPSD (Nano-Particle Shape Distribution)
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

# Single image with automatic OCR detection 
python3 main.py --mode single --input sample.tif --scale -1 --algo classical --min-size 3 --ocr-backend easyocr-auto

# Batch processing with auto OCR
python3 main.py --mode batch --input ./images/ --scale -1 --algo classical --min-size 5 --ocr-backend easyocr-auto

For detailed help:
python3 main.py --help
"""

# Standard library imports
import os
import sys


def show_usage_examples() -> None:
    """
    Display comprehensive usage examples when script is run without arguments.
    """
    print("\n" + "=" * 80)
    print("NanoPSD - Nano-Particle Shape Distribution Analyzer")
    print("=" * 80)
    print("\nUsage: python3 nanopsd.py [OPTIONS]")
    print("\nYou must provide command-line arguments. Here are common examples:\n")

    print("─" * 80)
    print("📋 BASIC EXAMPLES")
    print("─" * 80)

    print("\n1️⃣  Single image with manual scale bar value:")
    print("   python3 nanopsd.py --mode single --input sample.tif \\")
    print("                      --scale-bar-nm 200 --algo classical --min-size 3")

    print("\n2️⃣  Single image without scale bar (manual calibration):")
    print("   python3 nanopsd.py --mode single --input sample.tif \\")
    print("                      --nm-per-pixel 2.5 --algo classical --min-size 3")

    print("\n3️⃣  Batch processing with manual scale:")
    print("   python3 nanopsd.py --mode batch --input ./images/ \\")
    print("                      --scale-bar-nm 200 --algo classical --min-size 3")

    print("\n4️⃣  Batch processing without scale bars:")
    print("   python3 nanopsd.py --mode batch --input ./images/ \\")
    print("                      --nm-per-pixel 1.8 --algo classical --min-size 3")

    print("\n" + "─" * 80)
    print("🔍 OCR-BASED SCALE BAR DETECTION")
    print("─" * 80)

    print("\n5️⃣  Auto-detect with GPU (with CPU fallback option):")
    print("   python3 nanopsd.py --mode single --input sample.tif \\")
    print("                      --scale-bar-nm -1 --ocr-backend easyocr-auto \\")
    print("                      --algo classical --min-size 3")

    print("\n6️⃣  Auto-detect scale bar with EasyOCR (CPU forced):")
    print("   python3 nanopsd.py --mode single --input sample.tif \\")
    print("                      --scale-bar-nm -1 --ocr-backend easyocr-cpu \\")
    print("                      --algo classical --min-size 3")

    print("\n" + "─" * 80)
    print("✅ SCALE BAR VERIFICATION")
    print("─" * 80)

    print("\n7️⃣  Manual verification of detected scale bar:")
    print("   python3 nanopsd.py --mode single --input sample.tif \\")
    print("                      --ocr-backend easyocr-auto \\")
    print("                      --verify-scale-bar --algo classical --min-size 3")

    print("\n" + "─" * 80)
    print("📊 BATCH PROCESSING EXAMPLES")
    print("─" * 80)

    print("\n8️⃣  Batch with auto-detection (each image detected separately):")
    print("   python3 nanopsd.py --mode batch --input ./images/ \\")
    print("                      --ocr-backend easyocr-auto \\")
    print("                      --algo classical --min-size 5")

    print("\n9️⃣ Batch with verification prompts:")
    print("   python3 nanopsd.py --mode batch --input ./images/ \\")
    print("                      --ocr-backend easyocr-auto \\")
    print("                      --verify-scale-bar --algo classical --min-size 3")

    print("\n" + "─" * 80)
    print("⚙️  PARAMETER REFERENCE")
    print("─" * 80)

    print("\n  --mode            : 'single' or 'batch'")
    print("  --input             : Image path (single) or folder path (batch)")
    print("  --scale-bar-nm      : Scale bar size in nm")
    print("  --nm-per-pixel      : Direct calibration (for images without scale bars)")
    print("  --algo              : Segmentation algorithm (currently: 'classical')")
    print("  --min-size          : Minimum particle size in pixels (e.g., 3, 5)")
    print(
        "  --ocr-backend       : 'easyocr-auto' (default - Try GPU and fall back to GPU) or 'easyocr-cpu' (force CPU)"
    )
    print("  --verify-scale-bar  : Enable manual verification of scale detection")

    print("\n" + "─" * 80)
    print("💡 TIPS")
    print("─" * 80)

    print("\n  • Fastest: Use --scale-bar-nm with known value (no OCR)")
    print("  • No scale bar: Use --nm-per-pixel with calibration factor")
    print("  • CPU systems: Use --ocr-backend easyocr-cpu")
    print("  • GPU systems: Use --ocr-backend easyocr-auto")
    print("  • Uncertain detection: Add --verify-scale-bar flag")
    print(
        "  • Batch mode: All images must have same calibration if using --nm-per-pixel"
    )

    print("\n" + "─" * 80)
    print("📖 FULL HELP")
    print("─" * 80)

    print("\n  For complete parameter documentation, run:")
    print("  python3 nanopsd.py --help")

    print("\n" + "=" * 80 + "\n")


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

    # Local imports: Import heavy dependencies only when running analysis (lazy import)
    from scripts.cli import parse_args
    from pipeline.analyzer import NanoparticleAnalyzer

    # -------------------------------------------------------------------------
    # Step 1: Parse command-line arguments
    # -------------------------------------------------------------------------
    # This delegates to scripts/cli.py which defines all CLI options
    # If arguments are invalid, argparse will print help and exit automatically
    args = parse_args()

    # Validate and get morphology thresholds
    from scripts.cli import validate_morphology_thresholds

    thresholds = validate_morphology_thresholds(args)

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
        scale_bar_nm=args.scale_bar_nm,  # Physical scale in nm 
        nm_per_pixel=args.nm_per_pixel,  # Direct nm/pixel input
        # Segmentation parameters
        mode=args.algo,  # "classical" (only one implemented now)
        min_size_px=args.min_size,  # Minimum particle size filter (pixels)
        max_size_px=args.max_size,  # Maximum particle size filter (pixels)
        # OCR configuration
        ocr_backend=args.ocr_backend,  # "easyocr-auto" or "easyocr-cpu"
        verify_scale_bar=args.verify_scale_bar,  # Enable manual verification of scale detection
        save_preprocessing_steps=args.save_preprocessing_steps,
        save_segmentation_steps=args.save_segmentation_steps,
        # Morphology classification thresholds
        spherical_ar_max=thresholds["spherical_ar_max"],
        rodlike_ar_min=thresholds["rodlike_ar_min"],
        aggregate_c_max=thresholds["aggregate_c_max"],
        spherical_c_min=thresholds["spherical_c_min"],
        rodlike_s_min=thresholds["rodlike_s_min"],
        aggregate_s_max=thresholds["aggregate_s_max"],
        spherical_s_min=thresholds["spherical_s_min"],
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
    """
    import sys

    # Handle different argument scenarios
    if len(sys.argv) == 1:
        # No arguments - show examples
        show_usage_examples()
    elif "--help" in sys.argv or "-h" in sys.argv:
        # Show examples for --help (instant, no heavy imports)
        show_usage_examples()
        print("\n" + "─" * 80)
        print("For detailed technical help, run: python3 nanopsd.py --help-full")
        print("─" * 80 + "\n")
    elif "--help-full" in sys.argv:
        # Show full argparse help (slower, loads parser)
        from scripts.cli import build_parser

        build_parser().print_help()
    else:
        # Normal execution
        main()
