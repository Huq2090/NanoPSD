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
OCR Utilities for Scale Bar Text Recognition
=============================================

This module provides robust Optical Character Recognition (OCR) capabilities
specifically tuned for reading scale bar annotations in microscopy images.

Key Features:
-------------
- Supports both Tesseract and EasyOCR backends with user selection
- Handles various text formats: "50 nm", "0.2 µm", "100nm", etc.
- Automatic unit conversion (µm → nm)
- Multiple preprocessing strategies to handle different image qualities
- Graceful degradation if OCR backends aren't installed

Design Philosophy:
------------------
OCR is OPTIONAL. If backends aren't available, the pipeline still works
by falling back to CLI --scale parameter or filename parsing.

Backend Selection:
------------------
Users can choose OCR engine via --ocr-backend flag:
- "auto" (default): Try EasyOCR first, fall back to Tesseract
- "easyocr": Use only EasyOCR (recommended for microscopy images)
- "tesseract": Use only Tesseract (faster but less accurate)
"""

import re
import cv2
import numpy as np
from typing import Optional, Tuple
import gc
import torch

# =============================================================================
# Backend Detection and Registry
# =============================================================================
# Detect which OCR engines are installed on this system.
# This allows the code to work even if one or both backends are missing.

_BACKENDS = []  # List of available backend names

try:
    import pytesseract

    _BACKENDS.append("tesseract")
except ImportError:
    pass  # Tesseract not installed, continue without it

try:
    import easyocr

    _BACKENDS.append("easyocr")
except ImportError:
    pass  # EasyOCR not installed, continue without it


def clear_gpu_memory():
    """Clear GPU memory to prevent leaks."""
    try:
        import gc
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except:
        pass


# =============================================================================
# Text Parsing Functions
# =============================================================================


def parse_scale_text(text: str) -> Optional[Tuple[float, str]]:
    """
    Parse OCR'd text and extract numeric value with physical unit.

    This function is highly tolerant of OCR errors and formatting variations
    to maximize success rate across different microscope text styles.

    Supported Formats:
    ------------------
    - Standard: "50 nm", "200 nm", "100nm"
    - Micrometer: "0.2 µm", "0.5um", "1.0 µm"
    - Case variations: "NM", "nm", "Nm" (all treated equally)
    - Spacing variations: "50nm", "50 nm", "50  nm" (all accepted)
    - Decimal formats: "0.2", "0.20", ".2" (all valid)

    OCR Error Handling:
    -------------------
    - Greek mu (μ) vs micro sign (µ) → normalized
    - Letter 'u' misread as µ → corrected
    - Letter 'O' misread as zero → corrected
    - European decimals (comma) → converted to period

    Parameters
    ----------
    text : str
        Raw OCR output string (may contain noise/errors)
        Examples: "50 nm", "0.2um", "garbage 100 NM text"

    Returns
    -------
    (value_nm, "nm") : tuple or None
        - value_nm: float, physical length in nanometers (always converted)
        - "nm": str, always "nm" (normalized unit)
        - None if parsing completely fails

    Examples
    --------
    >>> parse_scale_text("50 nm")
    (50.0, "nm")

    >>> parse_scale_text("0.2 µm")
    (200.0, "nm")

    >>> parse_scale_text("random text 100nm more text")
    (100.0, "nm")

    >>> parse_scale_text("garbage with no scale")
    None
    """
    # Early exit for empty/None input
    if not text:
        return None

    # Step 1: Normalize common OCR confusions
    t = text.strip()
    t = t.replace("μ", "µ")  # Greek mu (μ) → micro sign (µ)
    t = t.replace("u", "µ")  # Common OCR error: 'u' instead of 'µ'
    t = t.replace(",", ".")  # European decimal notation support
    t = t.replace("O", "0")  # Letter O → zero (frequent OCR mistake)

    # Step 2: Try multiple regex patterns (ordered from most to least specific)
    patterns = [
        # Pattern 1: Standard format with clear unit
        # Matches: "50.5 nm", "0.2 µm", "100nm"
        r"(\d+(?:\.\d+)?)\s*(nm|µm|um)\b",
        # Pattern 2: Case-insensitive variants
        # Matches: "50 NM", "0.2 UM", "100 Nm"
        r"(\d+(?:\.\d+)?)\s*([nN][mM]|[µuU][mM])\b",
        # Pattern 3: Split decimal (OCR sometimes adds spaces)
        # Matches: "0 . 2 um", "50 . 5 nm"
        r"(\d+)\s*\.?\s*(\d*)\s*(nm|µm|um|NM|UM)\b",
    ]

    # Step 3: Try each pattern until one succeeds
    for pattern in patterns:
        m = re.search(pattern, t, flags=re.IGNORECASE)
        if m:
            # Handle split decimal case (pattern 3)
            if len(m.groups()) == 3 and m.group(2):
                # Reconstruct decimal: "0" + "." + "2" → "0.2"
                val = float(f"{m.group(1)}.{m.group(2)}")
                unit = m.group(3).lower()
            else:
                # Standard case: direct numeric extraction
                val = float(m.group(1))
                unit = m.group(2).lower()

            # Step 4: Normalize unit to nanometers
            if unit in ("nm", "NM"):
                return (val, "nm")

            if unit in ("µm", "um", "UM"):
                # Convert micrometers to nanometers
                # 1 µm = 1000 nm
                return (val * 1000.0, "nm")

    # No valid pattern matched - parsing failed
    return None


# =============================================================================
# Image Preprocessing for OCR
# =============================================================================


def _preprocess_for_ocr(image: np.ndarray, strategy: str) -> np.ndarray:
    """
    Apply different preprocessing strategies to enhance text readability.

    Why Multiple Strategies?
    -------------------------
    Different microscopy images have vastly different characteristics:
    - High contrast images → simple threshold works
    - Uneven illumination → adaptive threshold needed
    - Noisy images → denoising required
    - Faint text → sharpening helps
    - Broken characters → morphological operations reconnect them

    By trying multiple strategies, we maximize OCR success rate across
    all microscope types and imaging conditions.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (uint8, values 0-255)
    strategy : str
        Preprocessing method to apply:

        'basic': Simple Otsu threshold
            - Fastest method
            - Works for high-contrast, clean images
            - Use case: Modern SEM with clear scale bars

        'adaptive': Adaptive threshold
            - Handles uneven illumination/shading
            - Good for images with gradients or vignetting
            - Use case: TEM images with variable brightness

        'morph': Morphological closing
            - Connects broken/fragmented characters
            - Fills small gaps in text
            - Use case: Low-resolution or degraded images

        'denoise': Non-local means denoising
            - Removes salt-and-pepper noise
            - Preserves edges better than Gaussian
            - Use case: Noisy images from older microscopes

        'sharpen': Edge sharpening
            - Enhances faint/low-contrast text
            - Makes thin characters more prominent
            - Use case: Washed-out or overexposed images

    Returns
    -------
    processed : np.ndarray
        Binary image (0 or 255) optimized for OCR

    Notes
    -----
    - All strategies include 3x upscaling (OCR works best on larger text)
    - Output is always binary (pure black and white)
    - Processing order matters: blur before threshold for best results
    """

    # Step 1: Upscale for better OCR accuracy
    # Why 3x? OCR engines work best when text is 30-50 pixels tall.
    # Most microscopy scale text is 10-20 pixels, so 3x brings it to optimal range.
    img = cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # Step 2: Apply strategy-specific preprocessing

    if strategy == "basic":
        # Simple pipeline: blur → threshold
        # Gaussian blur reduces high-frequency noise before thresholding
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Otsu's method automatically finds optimal threshold value
        # THRESH_BINARY: pixels > threshold become 255 (white)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif strategy == "adaptive":
        # Best for uneven lighting conditions
        # Median blur preserves edges better than Gaussian
        img = cv2.medianBlur(img, 3)

        # Adaptive threshold: threshold computed locally for each region
        # Instead of one global threshold, each pixel's threshold depends
        # on the brightness of its neighborhood
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian-weighted average
            cv2.THRESH_BINARY,
            11,  # Neighborhood size (must be odd number)
            2,  # Constant subtracted from weighted mean
        )

    elif strategy == "morph":
        # Good for broken or fragmented characters
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing = dilation followed by erosion
        # Effect: connects nearby white regions (useful for broken text)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    elif strategy == "denoise":
        # Heavy denoising for very noisy images
        # Non-local means: looks for similar patches across entire image
        # More effective than simple blurring, but slower
        img = cv2.fastNlMeansDenoising(
            img,
            None,
            h=10,  # Filter strength (higher = more smoothing)
            templateWindowSize=7,  # Size of patches to compare
            searchWindowSize=21,  # Size of area to search for similar patches
        )
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif strategy == "sharpen":
        # Enhances edges (good for faint/washed-out text)
        # Sharpening kernel: emphasizes center pixel, subtracts neighbors
        # Effect: makes edges more prominent
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct text skew/rotation for improved OCR accuracy.

    Why Deskewing Matters:
    ----------------------
    OCR accuracy drops significantly when text is rotated. Even 5-10 degree
    angles can reduce recognition rate by 50% or more. This function detects
    and corrects the rotation.

    Algorithm:
    ----------
    1. Find all foreground (white) pixels (the text)
    2. Fit minimum area rectangle around them
    3. Extract rotation angle from rectangle orientation
    4. Rotate image to make text horizontal

    Parameters
    ----------
    image : np.ndarray
        Binary image with text (0 = background, >0 = text)

    Returns
    -------
    deskewed : np.ndarray
        Rotated image with horizontal text

    Notes
    -----
    - Only rotates if angle > 0.5 degrees (avoid unnecessary interpolation)
    - Uses high-quality cubic interpolation to minimize artifacts
    - Replicates borders to avoid black edges after rotation
    """
    # Find coordinates of all foreground pixels
    # np.where returns (rows, cols) of all pixels where condition is True
    coords = np.column_stack(np.where(image > 0))

    if len(coords) < 5:
        # Not enough pixels to reliably estimate angle
        # Need at least 5 points to fit a rectangle
        return image

    # Fit minimum area rectangle to the text pixels
    # This gives us the dominant orientation of the text
    # Returns: ((center_x, center_y), (width, height), angle)
    angle = cv2.minAreaRect(coords)[-1]

    # OpenCV's minAreaRect returns angle in range [-90, 0)
    # We need to correct it to [-45, 45) for proper text orientation
    if angle < -45:
        angle = 90 + angle

    # Only rotate if skew is significant
    # Small rotations introduce interpolation artifacts without much benefit
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Compute 2D rotation matrix
        # Parameters: center point, angle (counterclockwise), scale
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        # INTER_CUBIC: high-quality interpolation (slower but better quality)
        # BORDER_REPLICATE: extend edge pixels to avoid black borders
        image = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    return image


# =============================================================================
# Main OCR Interface
# =============================================================================


def ocr_read_number(
    image, lang_hint: str = "en", debug_dir: Optional[str] = None, backend: str = "auto"
) -> Optional[str]:
    """
    Perform OCR using selected backend with multiple preprocessing attempts.

    This is the main OCR entry point. It orchestrates multiple strategies
    to maximize success rate across different image types and qualities.

    Overall Strategy:
    -----------------
    1. Validate backend selection
    2. Try EasyOCR first (if auto or explicitly requested)
    3. Fall back to Tesseract (if auto or explicitly requested)
    4. For each backend, systematically try:
       - Normal and inverted images (black-on-white vs white-on-black)
       - Multiple preprocessing strategies (5 methods)
       - Deskewing (Tesseract only, EasyOCR handles rotation internally)
       - Multiple PSM modes (Tesseract only)
    5. Return immediately on first success (no need to try all combinations)

    Parameters
    ----------
    image : np.ndarray
        Grayscale image crop containing text (uint8, 0-255)
    lang_hint : str, optional
        Language hint for OCR (default: "eng" for English)
        Other options: 'fra', 'deu', 'spa', etc.
    debug_dir : str, optional
        If provided, save all preprocessing attempts to this folder
        Useful for debugging OCR failures
        Each saved image shows: backend_invert_strategy.png
    backend : str, optional
        OCR backend selection:

        "auto" (default): Try EasyOCR first, fall back to Tesseract
            - Best for most users
            - Maximizes success rate
            - Slightly slower (tries both if first fails)

        "easyocr": Use only EasyOCR
            - Most accurate for microscopy images
            - Handles rotated text automatically
            - Slower than Tesseract
            - Requires: pip install easyocr

        "tesseract": Use only Tesseract
            - Faster than EasyOCR
            - Good for high-contrast, horizontal text
            - Less accurate for complex backgrounds
            - Requires: apt-get install tesseract-ocr (+ pytesseract)

    Returns
    -------
    text : str or None
        Raw OCR output if successful
        Must contain at least one digit to be considered valid
        None if all attempts fail or no backend available

    Examples
    --------
    >>> text = ocr_read_number(crop, backend="auto")
    >>> if text:
    >>>     parsed = parse_scale_text(text)  # Get numeric value

    Notes
    -----
    - Returns on FIRST successful OCR (doesn't exhaust all attempts)
    - Only returns text containing digits (filters out pure noise)
    - Prints progress messages to help debug failures
    - Can be slow (especially EasyOCR on first run due to model loading)
    """

    # -------------------------------------------------------------------------
    # Step 1: Validate prerequisites and backend selection
    # -------------------------------------------------------------------------

    if not _BACKENDS:
        # No OCR backend installed at all
        print("⚠ No OCR backends available")
        print("  Install EasyOCR: pip install easyocr")
        print(
            "  Install Tesseract: apt-get install tesseract-ocr && pip install pytesseract"
        )
        return None

    # Validate backend parameter
    if backend not in ["auto", "easyocr", "tesseract"]:
        print(f"⚠ Invalid backend '{backend}', using 'auto'")
        backend = "auto"

    # Check if requested backend is actually available
    if backend == "easyocr" and "easyocr" not in _BACKENDS:
        print("⚠ EasyOCR requested but not installed")
        print("  Trying Tesseract instead")
        backend = "tesseract"
    elif backend == "tesseract" and "tesseract" not in _BACKENDS:
        print("⚠ Tesseract requested but not installed")
        print("  Trying EasyOCR instead")
        backend = "easyocr"

    # -------------------------------------------------------------------------
    # Step 2: Define strategies and parameters
    # -------------------------------------------------------------------------

    # Preprocessing strategies (in order of speed: fast → slow)
    strategies = ["basic", "adaptive", "morph", "denoise", "sharpen"]

    # Tesseract Page Segmentation Modes (PSM)
    # Different modes assume different text layouts
    psm_modes = [
        7,  # Single text line (best for scale bars - horizontal text)
        8,  # Single word (if text is just "200nm" with no spaces)
        6,  # Uniform block of text (fallback for multi-line)
        13,  # Raw line without layout analysis (last resort)
    ]

    # -------------------------------------------------------------------------
    # Step 3: Try EasyOCR (if auto or explicitly requested)
    # -------------------------------------------------------------------------

    if (backend == "auto" or backend == "easyocr") and "easyocr" in _BACKENDS:
        try:
            import easyocr

            print("  → Trying EasyOCR...")

            # Clear GPU memory before starting
            clear_gpu_memory()

            # Try GPU first, fallback to CPU if OOM
            reader = None
            try:
                reader = easyocr.Reader([lang_hint], gpu=True, verbose=False)
                print("    → Using GPU")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("    → GPU OOM, using CPU instead")
                    clear_gpu_memory()
                    reader = easyocr.Reader([lang_hint], gpu=False, verbose=False)
                else:
                    raise

            # Try both polarities
            for invert in [False, True]:
                img = image if not invert else (255 - image)

                for strategy in ["basic", "adaptive", "morph"]:
                    processed = _preprocess_for_ocr(img, strategy)

                    if debug_dir:
                        import os

                        os.makedirs(debug_dir, exist_ok=True)
                        tag = f"easyocr_inv{int(invert)}_{strategy}"
                        cv2.imwrite(f"{debug_dir}/prep_{tag}.png", processed)

                    results = reader.readtext(processed, detail=0)

                    if results:
                        txt = " ".join(results).strip()

                        if txt and any(c.isdigit() for c in txt):
                            print(f"    ✓ EasyOCR success: '{txt}'")
                            clear_gpu_memory()  # Clean up before returning
                            return txt

            # Clean up if no results
            clear_gpu_memory()

        except Exception as e:
            print(f"    ✗ EasyOCR error: {e}")
            clear_gpu_memory()
            if backend == "easyocr":
                return None

    # -------------------------------------------------------------------------
    # Step 4: Try Tesseract (if auto or explicitly requested)
    # -------------------------------------------------------------------------

    if (backend == "auto" or backend == "tesseract") and "tesseract" in _BACKENDS:
        try:
            import pytesseract

            print("  → Trying Tesseract...")

            # Try both polarities
            for invert in [False, True]:
                img = image if not invert else (255 - image)

                # Pre-compute deskewed version
                # Tesseract is sensitive to rotation, so correct it upfront
                img_deskewed = _deskew(img.copy())

                # Try all preprocessing strategies
                for strategy in strategies:
                    # Try both with and without deskewing
                    for use_deskew in [False, True]:
                        test_img = img_deskewed if use_deskew else img
                        processed = _preprocess_for_ocr(test_img, strategy)

                        # Save debug image if requested
                        if debug_dir:
                            import os

                            os.makedirs(debug_dir, exist_ok=True)
                            tag = f"tess_inv{int(invert)}_desk{int(use_deskew)}_{strategy}"
                            cv2.imwrite(f"{debug_dir}/prep_{tag}.png", processed)

                        # Try multiple PSM modes
                        for psm in psm_modes:
                            try:
                                # Configure Tesseract
                                config = (
                                    f"--oem 3 "  # OCR Engine Mode 3: Default (LSTM)
                                    f"--psm {psm} "  # Page segmentation mode
                                    "-c tessedit_char_whitelist=0123456789nmµuUM.NM "
                                    # Whitelist: only recognize these characters
                                    # Significantly improves accuracy by rejecting impossible chars
                                )

                                # Run Tesseract OCR
                                txt = pytesseract.image_to_string(
                                    processed, config=config, lang=lang_hint
                                ).strip()

                                # Validate: must contain digit
                                if txt and any(c.isdigit() for c in txt):
                                    print(f"    ✓ Tesseract success: '{txt}'")
                                    return txt

                            except Exception:
                                # This particular PSM mode failed
                                # Try next mode
                                continue

        except Exception as e:
            print(f"    ✗ Tesseract error: {e}")

    # -------------------------------------------------------------------------
    # Step 5: All attempts exhausted
    # -------------------------------------------------------------------------

    print("    ✗ All OCR attempts failed")
    return None
