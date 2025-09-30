"""
Command-line argument parser for NanoPSD.
Keep CLI concerns separate from the main pipeline entrypoint.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="NanoPSD", description="Nanoparticle Size Distribution Analyzer (SEM/TEM)"
    )
    p.add_argument(
        "--mode",
        choices=["single", "batch"],
        required=True,
        help="Run on a single image or a folder of images.",
    )
    p.add_argument(
        "--input", required=True, help="Path to input image (single) or folder (batch)."
    )
    p.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Known scale bar length in nanometers (e.g., 200).",
    )
    p.add_argument(
        "--algo",
        default="classical",
        choices=["classical", "ai", "both", "compare"],
        help="Segmentation/analysis algorithm family (default: classical).",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=3,
        help="Minimum particle size in pixels to keep after segmentation (default: 3).",
    )
    p.add_argument(
        "--ocr",
        choices=["auto", "easyocr", "tesseract"],
        default="auto",
        help="OCR backend: 'easyocr' (GPU-capable), 'tesseract', or 'auto' (prefer easyocr).",
    )
    return p


def parse_args():
    return build_parser().parse_args()
