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
    python3 main.py --mode single --input image.tif --scale-bar-nm -1 --algo classical --min-size 3

    # Single image with EasyOCR specifically
    python3 main.py --mode single --input image.tif --scale-bar-nm -1 --algo classical --min-size 3 --ocr-backend easyocr

    # Batch processing with OCR
    python3 main.py --mode batch --input ./images/ --scale-bar-nm -1 --algo classical --min-size 5 --ocr-backend auto
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
            "  Single image (OCR auto-detect):  python3 main.py --mode single --input image.tif --scale-bar-nm -1\n"
            "  Batch processing:                python3 main.py --mode batch --input ./folder/ --scale-bar-nm -1\n"
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
        required=True,
        metavar="VALUE",
        help=(
            "Scale bar length in nanometers:\n"
            "  Positive value (e.g., 200): Use this as the known scale\n"
            "  -1: Attempt automatic OCR detection of scale bar text\n"
            "\n"
            "Examples:\n"
            "  --scale-bar-nm 200    # Scale bar is 200 nm\n"
            "  --scale-bar-nm 0.5    # Scale bar is 0.5 nm (rare, but supported)\n"
            "  --scale-bar-nm -1     # Auto-detect using OCR"
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
        "--ocr-backend",
        default="auto",
        choices=["auto", "easyocr", "tesseract"],
        metavar="BACKEND",
        help=(
            "OCR engine for scale bar text recognition (default: auto)\n"
            "\n"
            "Backend Options:\n"
            "  'auto':      Try EasyOCR first, fall back to Tesseract if needed\n"
            "               (Recommended for most users - maximizes success rate)\n"
            "\n"
            "  'easyocr':   Use only EasyOCR (deep learning-based)\n"
            "               + Most accurate for microscopy images\n"
            "               + Handles rotated/skewed text automatically\n"
            "               + Works with complex backgrounds\n"
            "               - Slower (1-2 seconds per image)\n"
            "               - Requires: pip install easyocr\n"
            "\n"
            "  'tesseract': Use only Tesseract (traditional OCR)\n"
            "               + Faster (~0.1 seconds per image)\n"
            "               + Good for high-contrast, horizontal text\n"
            "               - Less accurate for complex images\n"
            "               - Requires: apt-get install tesseract-ocr\n"
            "                          pip install pytesseract\n"
            "\n"
            "Note: This flag only matters when --scale-bar-nm -1 (OCR mode).\n"
            "      When providing manual scale (e.g., --scale-bar-nm 200), OCR is not used."
        ),
    )

    p.add_argument(
        "--verify-scale-bar",
        action="store_true",
        help="Show detected scale bar and wait for user confirmation (Y/N) before processing",
    )
    return p


def parse_args():
    """
    Parse command-line arguments from sys.argv.

    This is a thin wrapper around build_parser().parse_args() for convenience.
    By separating parser construction from parsing, we make testing easier
    (can construct parser without immediately parsing sys.argv).

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments as an object with attributes matching argument names

        Example attributes:
        - args.mode: "single" or "batch"
        - args.input: path to image or folder
        - args.scale: float (scale bar length in nm, or -1 for OCR)
        - args.algo: segmentation algorithm choice
        - args.min_size: minimum particle size in pixels
        - args.ocr_backend: "auto", "easyocr", or "tesseract"

    Raises
    ------
    SystemExit
        If required arguments are missing or invalid values provided
        (argparse handles this automatically with helpful error messages)

    Example Usage
    -------------
    >>> from scripts.cli import parse_args
    >>> args = parse_args()
    >>> print(f"Processing: {args.input}")
    >>> print(f"Scale: {args.scale} nm")
    >>> print(f"OCR Backend: {args.ocr_backend}")
    """
    return build_parser().parse_args()
