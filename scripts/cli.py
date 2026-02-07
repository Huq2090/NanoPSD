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
Command-Line Interface (CLI) Parser for NanoPSD
================================================

This module handles all command-line argument parsing for the NanoPSD pipeline.
By keeping CLI logic separate from the main pipeline code, we maintain clean
separation of concerns and make the codebase easier to maintain.

Design Principles:
------------------
- User-friendly: Clear help messages and sensible defaults
- Flexible: Support both single-image and batch processing
- Documented: Each argument has detailed help text
- Extensible: Easy to add new arguments without touching pipeline code
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the argument parser with all NanoPSD command-line options.

    This function defines the complete CLI interface. Each argument includes:
    - Short description (for --help output)
    - Type validation (int, float, choice list, etc.)
    - Default values (where applicable)
    - Required/optional status

    Returns
    -------
    parser : argparse.ArgumentParser
        Configured argument parser ready to parse sys.argv

    Usage Examples
    --------------
    # Single image with known scale (manual)
    python3 main.py --mode single --input image.tif --scale-bar-nm 200 --algo classical --min-size 3

    # Single image with OCR auto-detection
    python3 main.py --mode single --input image.tif --algo classical --min-size 3 --ocr-backend auto

    # Single image with EasyOCR specifically
    python3 main.py --mode single --input image.tif --algo classical --min-size 3 --ocr-backend easyocr

    # Batch processing with OCR
    python3 main.py --mode batch --input ./images/ --algo classical --min-size 5 --ocr-backend auto
    """

    # Create parser with program description
    p = argparse.ArgumentParser(
        prog="NanoPSD",
        description=(
            "Nanoparticle Size Distribution Analyzer for SEM/TEM Images\n\n"
            "Automatically detects scale bars, segments nanoparticles, and "
            "generates size distribution statistics from electron microscopy images."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Single image (manual scale):     python3 main.py --mode single --input image.tif --scale-bar-nm 200\n"
            "  Single image (OCR auto-detect):  python3 main.py --mode single --input image.tif --ocr-backend auto\n"
            "  Batch processing:                python3 main.py --mode batch --input ./folder/ --ocr-backend auto\n"
            "\n"
            "For more information, visit: https://github.com/Huq2090/NanoPSD"
        ),
    )

    # Required Arguments
    p.add_argument(
        "--mode",
        choices=["single", "batch"],
        required=True,
        help=(
            "Processing mode:\n"
            "  'single' = analyze one image\n"
            "  'batch'  = analyze all images in a folder"
        ),
    )

    p.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help=(
            "Input path:\n"
            "  For single mode: path to image file (e.g., 'sample.tif')\n"
            "  For batch mode:  path to folder (e.g., './images/')"
        ),
    )

    p.add_argument(
        "--scale-bar-nm",
        type=float,
        required=False,
        default=None,
        metavar="VALUE",
        help=(
            "Scale bar length in nanometers (manual mode):\n"
            "  Provide the known scale bar value from the image.\n"
            "\n"
            "Examples:\n"
            "  --scale-bar-nm 200    # Scale bar is 200 nm\n"
            "  --scale-bar-nm 0.5    # Scale bar is 0.5 nm\n"
            "\n"
            "Note: For automatic detection, use --ocr-backend instead.\n"
            "      Cannot be used together with --ocr-backend or --nm-per-pixel."
        ),
    )

    # Optional Arguments
    p.add_argument(
        "--algo",
        default="classical",
        choices=["classical", "ai", "both", "compare"],
        metavar="ALGORITHM",
        help=(
            "Segmentation algorithm (default: classical):\n"
            "  'classical' = Otsu thresholding (fast, works for most images)\n"
            "  'ai'        = Deep learning segmentation (not yet implemented)\n"
            "  'both'      = Run both and use AI results (not yet implemented)\n"
            "  'compare'   = Run both and generate comparison report (not yet implemented)\n"
            "\n"
            "Note: Only 'classical' is currently available."
        ),
    )

    # ADD new --nm-per-pixel argument
    p.add_argument(
        "--nm-per-pixel",
        type=float,
        required=False,
        default=None,
        metavar="VALUE",
        help=(
            "Direct calibration for images WITHOUT scale bars.\n"
            "Provide the conversion factor: nanometers per pixel.\n"
            "\n"
            "Use this when:\n"
            "  - Image has no scale bar\n"
            "  - You know the exact nm/pixel value from microscope settings\n"
            "\n"
            "Example:\n"
            "  --nm-per-pixel 2.5\n"
            "\n"
            "NOTE: Use EITHER --scale-bar-nm OR --nm-per-pixel (not both)\n"
        ),
    )

    p.add_argument(
        "--min-size",
        type=int,
        default=3,
        metavar="PIXELS",
        help=(
            "Minimum particle size in pixels (default: 3)\n"
            "\n"
            "Particles smaller than this are filtered out as noise.\n"
            "Typical values:\n"
            "  3-5:  For high-resolution images (good signal-to-noise)\n"
            "  5-10: For noisy images or lower magnification\n"
            "  10+:  When you only want to analyze larger particles\n"
            "\n"
            "Lower values = more particles detected (including noise)\n"
            "Higher values = fewer false positives, but may miss small particles"
        ),
    )

    p.add_argument(
        "--max-size",
        type=int,
        default=None,
        metavar="PIXELS",
        help=(
            "Maximum particle size in pixels (default: None = no limit)\n"
            "\n"
            "Particles larger than this are filtered out.\n"
            "Useful for removing large false detections or artifacts.\n"
            "\n"
            "Example: --max-size 100\n"
        ),
    )

    p.add_argument(
        "--ocr-backend",
        default=None,
        choices=["auto", "tesseract", "easyocr"],
        metavar="BACKEND",
        help=(
            "Enable automatic scale bar detection using OCR:\n"
            "\n"
            "Backend Options:\n"
            "  'auto':      Try EasyOCR first, fallback to Tesseract (recommended)\n"
            "               Maximizes detection success rate\n"
            "\n"
            "  'tesseract': Traditional OCR (CPU-friendly)\n"
            "               + Faster (~0.1 seconds per image)\n"
            "               + Good for high-contrast, horizontal text\n"
            "               - Less accurate for rotated/skewed text\n"
            "               Requires: sudo apt-get install tesseract-ocr\n"
            "                        pip install pytesseract\n"
            "\n"
            "  'easyocr':   Deep learning OCR (GPU recommended)\n"
            "               + Most accurate for microscopy images\n"
            "               + Handles rotated/skewed text automatically\n"
            "               + Works with complex backgrounds\n"
            "               - Slower (1-2 seconds per image)\n"
            "               - WARNING: Very slow on CPU (hours vs seconds)\n"
            "               Requires: pip install easyocr torch torchvision\n"
            "\n"
            "Examples:\n"
            "  --ocr-backend auto        # Auto-detect with EasyOCR first\n"
            "  --ocr-backend tesseract    # Auto-detect with Tesseract\n"
            "  --ocr-backend easyocr      # Auto-detect with EasyOCR\n"
            "\n"
            "Note: Cannot be used together with --scale-bar-nm or --nm-per-pixel.\n"
            "      Choose ONE calibration method only."
        ),
    )

    p.add_argument(
        "--verify-scale-bar",
        action="store_true",
        help="Show detected scale bar and wait for user confirmation (Y/N) before processing",
    )

    p.add_argument(
        "--save-preprocessing-steps",
        action="store_true",
        help=(
            "Save intermediate preprocessing step images for visualization.\n"
            "Useful for papers, presentations, and debugging.\n"
            "Images saved to: outputs/preprocessing_steps/\n"
            "\n"
            "Steps saved:\n"
            "  1. Original image\n"
            "  2. Normalized (0-255)\n"
            "  3. CLAHE enhanced\n"
            "  4. Gaussian blurred\n"
            "  5. Otsu thresholded\n"
            "  6. Inverted binary mask"
        ),
    )

    p.add_argument(
        "--save-segmentation-steps",
        action="store_true",
        help=(
            "Save intermediate segmentation step images for visualization.\n"
            "Useful for papers, presentations, and debugging.\n"
            "Images saved to: outputs/segmentation_steps/\n"
            "\n"
            "Steps saved:\n"
            "  1. Input binary mask (from preprocessing)\n"
            "  2. After small object removal (min-size filter)\n"
            "  3. After large object removal (max-size filter, if used)\n"
            "  4. After hole filling\n"
            "  5. Labeled components (color-coded particles)"
        ),
    )

    return p


def parse_args():
    parser = build_parser()
    args = parser.parse_args()

    # Check which calibration methods are provided
    has_scale_bar = args.scale_bar_nm is not None
    has_ocr = args.ocr_backend is not None
    has_nm_per_px = args.nm_per_pixel is not None

    # Count how many methods provided
    methods_count = sum([has_scale_bar, has_ocr, has_nm_per_px])

    # Must provide exactly ONE method
    if methods_count == 0:
        parser.error(
            "Must provide ONE calibration method:\n"
            "  Option 1: --scale-bar-nm VALUE     (manual scale value)\n"
            "  Option 2: --ocr-backend BACKEND    (automatic OCR detection)\n"
            "  Option 3: --nm-per-pixel VALUE     (no scale bar, direct calibration)\n"
            "\n"
            "Examples:\n"
            "  python3 nanopsd.py --input image.tif --scale-bar-nm 200 --min-size 3\n"
            "  python3 nanopsd.py --input image.tif --ocr-backend tesseract --min-size 3\n"
            "  python3 nanopsd.py --input image.tif --nm-per-pixel 2.5 --min-size 3"
        )

    if methods_count > 1:
        methods_used = []
        if has_scale_bar:
            methods_used.append("--scale-bar-nm")
        if has_ocr:
            methods_used.append("--ocr-backend")
        if has_nm_per_px:
            methods_used.append("--nm-per-pixel")

        parser.error(
            f"Cannot use multiple calibration methods together.\n"
            f"You provided: {', '.join(methods_used)}\n"
            f"\n"
            f"Choose ONE of:\n"
            f"  --scale-bar-nm (manual scale value)\n"
            f"  --ocr-backend (automatic OCR)\n"
            f"  --nm-per-pixel (no scale bar)"
        )

    return args
