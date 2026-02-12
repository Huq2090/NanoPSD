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
Scale Bar Detection and OCR for Microscopy Images
==================================================

This module handles automatic detection and measurement of scale bars
in electron microscopy (SEM/TEM) images, plus OCR of the associated text labels.

Architecture:
-------------
1. Geometric detection (fast, robust, microscope-agnostic)
   - Locates the scale bar using shape analysis
   - Returns bar width in pixels + bounding box + mask

2. OCR text recognition (optional, enhances automation)
   - Reads "50 nm", "0.2 µm", etc. from the image
   - Converts units to nanometers automatically
   - Falls back to CLI parameter or filename if OCR fails

Key Design Decisions:
---------------------
- Geometry-first approach: works even without OCR backends
- Multi-polarity thresholding: handles both dark-on-light and light-on-dark bars
- Hough line fallback: catches thin "hairline" bars missed by blob detection
- Wide OCR search: looks in multiple regions (corners, full bottom strip)
- Filename parsing fallback: extracts scale from filenames like "sample_200nm.tif"
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import os
import re

# Try to import OCR utilities (graceful degradation if unavailable)
try:
    from .ocr import ocr_read_number, parse_scale_text
except Exception:
    # Fallback stubs if OCR module isn't available
    def ocr_read_number(
        image, lang_hint: str = "eng", debug_dir=None, backend: str = "auto"
    ):
        """Stub: OCR unavailable, returns None."""
        return None

    def parse_scale_text(text: str):
        """Stub: OCR unavailable, returns None."""
        return None


# =============================================================================
# Helper Functions for Scale Bar Geometry Detection
# =============================================================================


def _bottom_band_roi(
    image: np.ndarray, frac: float = 0.25
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract bottom region of image where scale bars typically appear.

    Motivation:
    -----------
    Scale bars are almost always in the bottom portion of microscopy images.
    By restricting our search to this region, we:
    - Reduce false positives (fewer non-bar objects to consider)
    - Speed up processing (smaller region to analyze)
    - Improve robustness (avoid confusing particles/features as bars)

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (H x W)
    frac : float, default=0.25
        Fraction of image height to keep from bottom
        (0.25 = bottom 25% of image)

    Returns
    -------
    roi : np.ndarray
        Cropped bottom band (smaller height x same width)
    roi_box : (x, y, w, h)
        Bounding box of ROI in original image coordinates
        x=0, y=top_of_bottom_band, w=full_width, h=band_height

    Example
    -------
    For a 1000x800 image with frac=0.25:
    - Returns roi of shape (200 x 800)  [bottom 200 rows]
    - roi_box = (0, 800, 800, 200)
    """
    h, w = image.shape[:2]

    # Clamp fraction to sensible range (5% to 50% of image)
    frac = max(0.05, min(frac, 0.5))

    # Calculate starting row for bottom band
    y0 = int(h * (1.0 - frac))

    # Extract bottom rows
    roi = image[y0:h, 0:w]

    return roi, (0, y0, w, h - y0)


def _threshold_and_candidates(roi_gray: np.ndarray):
    """
    Generate candidate contours by thresholding in both polarities.

    Why both polarities?
    --------------------
    Different microscope manufacturers use different rendering:
    - Some: BLACK bar on LIGHT background (Otsu on original)
    - Others: WHITE bar on DARK background (Otsu on inverted)

    By trying both, we handle all cases without prior knowledge.

    Parameters
    ----------
    roi_gray : np.ndarray
        Grayscale region of interest (bottom band)

    Returns
    -------
    results : list of (binary_mask, contours)
        Two entries: [normal_polarity, inverted_polarity]
        Each entry contains:
        - binary_mask: thresholded image (uint8)
        - contours: list of detected contour arrays

    Algorithm
    ---------
    For each polarity:
    1. Apply Otsu's threshold (automatic optimal threshold selection)
    2. Morphological opening (remove salt-and-pepper noise)
    3. Find external contours (bar is an outer object)
    """
    results = []

    # Try both polarities: normal and inverted
    for invert in (False, True):
        # Invert image for second iteration
        g = roi_gray if not invert else (255 - roi_gray)

        # Otsu threshold: automatically determines optimal threshold value
        # Separates foreground (potential bars) from background
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological opening: erosion followed by dilation
        # Removes small noise specks while preserving large structures (bars)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours (boundaries of white regions)
        # RETR_EXTERNAL: only outermost contours (ignores holes)
        # CHAIN_APPROX_SIMPLE: compress contour representation
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results.append((bw, contours))

    return results


def _score_bar_candidate(
    w: int,  # Width of bounding box
    h: int,  # Height of bounding box
    y_center: float,  # Vertical center position
    roi_h: int,  # ROI height (for normalization)
    solidity: float,  # Convexity measure (area / hull_area)
    extent: float,  # Rectangularity (area / bbox_area)
    dist_edge: float,  # Distance to nearest edge
) -> float:
    """
    Score a candidate blob based on how "scale-bar-like" it is.

    Scoring Intuition:
    ------------------
    Scale bars have distinctive geometric properties:
    1. HIGH aspect ratio (long and thin, typically 10:1 or more)
    2. HIGH solidity (solid shape, no internal holes)
    3. HIGH extent (fills its bounding box, rectangle-like)
    4. BOTTOM position (near bottom edge of ROI)
    5. CORNER/EDGE proximity (often hugs bottom-left or bottom-right)

    Parameters
    ----------
    w, h : int
        Bounding box dimensions in pixels
    y_center : float
        Vertical center of blob within ROI
    roi_h : int
        Total ROI height (for normalizing vertical position)
    solidity : float in [0, 1]
        Area / ConvexHullArea (1.0 = perfectly convex)
    extent : float in [0, 1]
        Area / BoundingBoxArea (1.0 = perfect rectangle)
    dist_edge : float
        Minimum distance to any edge (in pixels)

    Returns
    -------
    score : float
        Weighted composite score (higher = more bar-like)
        Range: approximately [0, 1]

    Weights
    -------
    45% - Aspect ratio (dominant feature)
    20% - Solidity (second most important)
    15% - Extent (rectangular shape)
    10% - Vertical position (prefer bottom)
    10% - Edge proximity (prefer corners)
    """

    # Aspect ratio: width / height (avoid division by zero)
    aspect = w / max(h, 1)

    # Ideal vertical position: ~90% down in ROI (near bottom)
    # Score decreases linearly as distance from 0.9 increases
    near_bottom = 1.0 - abs((y_center / max(roi_h, 1)) - 0.9)

    # Composite weighted score
    score = (
        0.45 * min(aspect / 12.0, 1.0)  # Saturate at aspect ratio 12
        + 0.20 * min(solidity, 1.0)  # Already normalized [0, 1]
        + 0.15 * min(extent / 0.85, 1.0)  # Target extent ~0.85
        + 0.10 * near_bottom  # Prefer bottom positions
        + 0.10 * (1.0 - min(dist_edge / 30.0, 1.0))  # Prefer within 30px of edge
    )

    return float(score)


def _mask_from_bbox(
    shape: Tuple[int, int], bbox: Tuple[int, int, int, int], pad: int = 2
) -> np.ndarray:
    """
    Create binary mask covering scale bar region (for exclusion from analysis).

    Purpose:
    --------
    After detecting the scale bar, we must EXCLUDE it from particle analysis.
    Otherwise, the bar itself would be counted as a giant "particle"!

    This function creates a binary mask that can be applied to zero-out
    the scale bar region during segmentation.

    Parameters
    ----------
    shape : (H, W)
        Full image dimensions
    bbox : (x, y, w, h)
        Scale bar bounding box in full-image coordinates
    pad : int, default=2
        Padding around bbox (ensures we mask adjacent text too)

    Returns
    -------
    mask : np.ndarray, shape=(H, W), dtype=uint8
        Binary mask: 0 everywhere, 255 in bar region

    Example
    -------
    >>> mask = _mask_from_bbox((1000, 800), (50, 950, 200, 10), pad=5)
    >>> # Mask is 255 in rectangle (45, 945) to (255, 965)
    >>> # Can be used: binary_image[mask > 0] = False
    """
    H, W = shape[:2]
    x, y, w, h = bbox

    # Apply padding (with boundary clipping)
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, W)
    y1 = min(y + h + pad, H)

    # Create blank mask
    mask = np.zeros((H, W), dtype=np.uint8)

    # Fill bar region with 255
    mask[y0:y1, x0:x1] = 255

    return mask


# =============================================================================
# Main Detection Functions
# =============================================================================
# def _detect_text_region(
#     image: np.ndarray, bar_bbox: Tuple[int, int, int, int]
# ) -> Optional[Tuple[int, int, int, int]]:
#     """
#     Detect text region near the scale bar.

#     Returns bbox (x, y, w, h) of text region, or None if not found.
#     """
#     x, y, w, h = bar_bbox
#     H, W = image.shape[:2]

#     # Search region: area to the right and below the bar
#     search_x1 = max(0, x - 50)
#     search_x2 = min(W, x + w + 200)
#     search_y1 = max(0, y - 30)
#     search_y2 = min(H, y + h + 50)

#     search_roi = image[search_y1:search_y2, search_x1:search_x2]

#     # Threshold to find text (usually darker than background)
#     _, binary = cv2.threshold(
#         search_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )

#     # Find contours of potential text
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     text_boxes = []
#     for cnt in contours:
#         tx, ty, tw, th = cv2.boundingRect(cnt)
#         aspect = tw / max(th, 1)

#         # Text has moderate aspect ratio (not too wide like bar)
#         if 0.3 < aspect < 5.0 and tw > 5 and th > 5:
#             # Convert to full image coordinates
#             full_x = search_x1 + tx
#             full_y = search_y1 + ty
#             text_boxes.append((full_x, full_y, tw, th))

#     if text_boxes:
#         # Merge all text boxes into one region
#         min_x = min(box[0] for box in text_boxes)
#         min_y = min(box[1] for box in text_boxes)
#         max_x = max(box[0] + box[2] for box in text_boxes)
#         max_y = max(box[1] + box[3] for box in text_boxes)

#         return (min_x, min_y, max_x - min_x, max_y - min_y)

#     return None


def _detect_text_near_bar(
    image: np.ndarray, bar_bbox: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect text region near scale bar.

    Returns:
        (x, y, w, h) bounding box of text, or None
    """
    x, y, w, h = bar_bbox
    H, W = image.shape[:2]

    # Search region: expand around bar
    search_x1 = max(0, x - 100)
    search_x2 = min(W, x + w + 150)
    search_y1 = max(0, y - 50)
    search_y2 = min(H, y + h + 50)

    roi = image[search_y1:search_y2, search_x1:search_x2]

    # Threshold to find dark text
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_candidates = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)

        # Text has moderate aspect ratio (not long like bar)
        if 0.2 < aspect < 8.0 and cw > 8 and ch > 8 and cw < w * 2:
            # Convert to full image coordinates
            full_x = search_x1 + cx
            full_y = search_y1 + cy
            text_candidates.append((full_x, full_y, cw, ch))

    if text_candidates:
        # Merge all text boxes
        min_x = min(box[0] for box in text_candidates)
        min_y = min(box[1] for box in text_candidates)
        max_x = max(box[0] + box[2] for box in text_candidates)
        max_y = max(box[1] + box[3] for box in text_candidates)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    return None


def detect_scale_bar(
    image_path: str,
    save_debug: bool = True,
    debug_dir: str = "outputs/figures",
) -> Tuple[int, Tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """
    Detect scale bar using hybrid geometry + Hough line approach.

    Detection Pipeline:
    -------------------
    1. Extract bottom-band ROI (where bars live)
    2. Threshold in both polarities (dark bar / light bar)
    3. Evaluate each blob using geometric scoring
    4. Keep highest-scoring candidate
    5. If no candidate found, fall back to Hough line detection
    6. Return bar width (pixels), bbox, mask, visualization

    Robustness Features:
    --------------------
    - Handles both dark-on-light and light-on-dark bars
    - Handles thick "blob" bars and thin "hairline" bars
    - Microscope-agnostic (no assumptions about specific formats)
    - Debug visualizations for troubleshooting

    Parameters
    ----------
    image_path : str
        Path to grayscale or color microscopy image
    save_debug : bool, default=True
        If True, save annotated images showing detection results
    debug_dir : str, default="outputs/figures"
        Folder for debug output images

    Returns
    -------
    width_px : int
        Detected scale bar length in pixels (width of bbox)
    bar_bbox : (x, y, w, h)
        Bounding box in full-image coordinates
    bar_mask : np.ndarray
        Binary mask (255 inside bar region, 0 elsewhere)
    roi_vis : np.ndarray
        BGR visualization of ROI with detected bar highlighted

    Raises
    ------
    ValueError
        If image cannot be read or no scale bar detected
    """

    # Step 0: Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Step 1: Extract bottom-band ROI (reduce search space)
    roi, (rx, ry, rw, rh) = _bottom_band_roi(img, frac=0.25)

    # Step 2: Generate and score candidate blobs
    best = None  # Will hold: (score, width_px, bbox_full, bar_mask, roi_vis)

    for bw, contours in _threshold_and_candidates(roi):
        for cnt in contours:
            if len(cnt) < 5:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            if ch < 1 or cw < 10:
                continue

            aspect = cw / max(ch, 1)
            if aspect < 5.0:
                continue

            # CRITICAL: Reject candidates that are too large (entire black strips)
            # Scale bar should be thin - reject if height is too large
            if ch > min(50, rh * 0.2):  # Max 50 pixels OR 20% of ROI height
                continue

            # Also reject if the bar is suspiciously wide (covers most of bottom)
            if cw > rw * 0.7:  # Wider than 70% of image width
                continue

            area = cv2.contourArea(cnt)
            rect_area = cw * ch
            extent = area / max(rect_area, 1)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)

            dist_edge = min(x, y, rw - (x + cw), rh - (y + ch))

            score = _score_bar_candidate(
                w=cw,
                h=ch,
                y_center=y + ch / 2.0,
                roi_h=rh,
                solidity=solidity,
                extent=extent,
                dist_edge=dist_edge,
            )

            bbox_full = (x + rx, y + ry, cw, ch)
            bar_mask = _mask_from_bbox(img.shape, bbox_full, pad=2)

            if (
                (best is None)
                or (score > best[0])
                or (score == best[0] and cw > best[1])
            ):
                roi_vis = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
                cv2.rectangle(roi_vis, (x, y), (x + cw, y + ch), (0, 255, 255), 2)
                best = (score, cw, bbox_full, bar_mask, roi_vis)

    # Step 3: Fallback to Hough line detection for thin bars
    if best is None:
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=max(roi.shape[1] // 8, 10),
            maxLineGap=10,
        )

        if lines is not None and len(lines) > 0:
            best_len = 0
            best_bbox = None

            for x1, y1, x2, y2 in lines[:, 0, :]:
                length = int(np.hypot(x2 - x1, y2 - y1))

                if length > best_len and abs(y2 - y1) <= 3:
                    best_len = length
                    x = min(x1, x2)
                    y = min(y1, y2)
                    best_bbox = (x + rx, y + ry, max(1, best_len), 3)

            if best_bbox is not None:
                width_px = int(best_bbox[2])
                bar_mask = _mask_from_bbox(img.shape, best_bbox, pad=2)

                roi_vis = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
                cv2.rectangle(
                    roi_vis,
                    (best_bbox[0] - rx, best_bbox[1] - ry),
                    (
                        best_bbox[0] - rx + best_bbox[2],
                        best_bbox[1] - ry + best_bbox[3],
                    ),
                    (0, 255, 0),
                    2,
                )

                if save_debug:
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(f"{debug_dir}/scale_candidates_hough.png", roi_vis)

                return width_px, best_bbox, bar_mask, roi_vis

        raise ValueError("Scale bar not detected.")

    # Step 4: Extract results from best blob candidate
    _, width_px, bbox_full, bar_mask, roi_vis = best

    # Step 5: Save debug visualizations
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

        dbg = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        x, y, w, h = bbox_full
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imwrite(f"{debug_dir}/scale_bar_final.png", dbg)

        cv2.imwrite(f"{debug_dir}/scale_candidates.png", roi_vis)

    # Detect text region and expand mask
    # text_bbox = _detect_text_region(img, bbox_full)
    # if text_bbox is not None:
    #     text_mask = _mask_from_bbox(img.shape, text_bbox, pad=5)
    #     bar_mask = cv2.bitwise_or(bar_mask, text_mask)  # Combine masks

    # Also detect and mask text region
    text_bbox = _detect_text_near_bar(img, bbox_full)
    if text_bbox is not None:
        text_mask = _mask_from_bbox(img.shape, text_bbox, pad=10)
        # Combine bar mask and text mask
        bar_mask = cv2.bitwise_or(bar_mask, text_mask)
        print(f"Excluded text region: {text_bbox}")

    return int(width_px), bbox_full, bar_mask, roi_vis


def detect_scale_label(
    image_path: str,
    bar_bbox: Tuple[int, int, int, int],
    save_debug: bool = True,
    debug_dir: str = "outputs/figures",
    ocr_backend: str = "easyocr-auto",  # NEW: Backend selection parameter
) -> Tuple[Optional[float], Optional[Tuple[int, int, int, int]]]:
    """
    Read scale bar text using OCR and convert to nanometers.

    Search Strategy:
    ----------------
    1. Look NEAR the bar (right, left, above)
    2. Search ENTIRE bottom strip in small chunks
    3. Check BOTTOM CORNERS specifically
    4. Fall back to FILENAME parsing (e.g., "sample_200nm.tif")

    OCR Preprocessing:
    ------------------
    For each region, ocr_read_number() tries multiple strategies based on backend.

    Parameters
    ----------
    image_path : str
        Path to the microscopy image
    bar_bbox : (x, y, w, h)
        Detected scale bar bounding box (from detect_scale_bar)
    save_debug : bool, default=True
        If True, save all searched regions for inspection
    debug_dir : str, default="outputs/figures"
        Folder for debug output
    ocr_backend : str, default="auto"
        OCR engine to use: "easyocr-auto" or "easyocr-cpu"

    Returns
    -------
    value_nm : float or None
        Physical scale bar length in nanometers
        None if OCR fails in all regions

    Supported Formats:
    ------------------
    - "50 nm", "100nm", "200 nm."
    - "0.2 µm", "0.5um", "1.0 µm"  (auto-converted to nm)
    - Filename patterns: "sample_200nm.tif", "image-0.5um.png"
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    x, y, w, h = bar_bbox
    H, W = img.shape[:2]

    # Helper: Parse scale from filename
    def _parse_from_filename(path: str) -> Optional[float]:
        name = os.path.basename(path).replace("μ", "µ").lower()

        m = re.search(r"(\d+(?:\.\d+)?)\s*nm\b", name)
        if m:
            return float(m.group(1))

        m = re.search(r"(\d+(?:\.\d+)?)\s*(µm|um)\b", name)
        if m:
            return float(m.group(1)) * 1000.0

        return None

    # Define search regions
    search_regions = []

    # Near the bar
    near_pads = [
        (w + 5, -2 * h, min(W, x + w + W // 3) - (x + w + 5), 3 * h),
        (-W // 3, -2 * h, w - 5 + W // 3, 3 * h),
        (-W // 3, -int(3 * h), W // 3 + w, int(4 * h)),
    ]

    for dx, dy, ww, hh in near_pads:
        rx = max(0, x + dx)
        ry = max(0, y + dy)
        rx2 = min(W, rx + int(ww))
        ry2 = min(H, ry + int(hh))
        if (rx2 - rx) > 8 and (ry2 - ry) > 8:
            search_regions.append((rx, ry, rx2, ry2))

    # Bottom strip in chunks
    by0 = int(H * 0.7)
    strip_h = H - by0
    chunk_w = max(W // 5, 100)

    for i in range(0, W, chunk_w):
        rx2 = min(i + chunk_w, W)
        if rx2 - i > 50:
            search_regions.append((i, by0, rx2, H))

    # Bottom corners
    corner_size = min(W // 4, 300)
    search_regions.append((0, by0, corner_size, H))
    search_regions.append((W - corner_size, by0, W, H))

    # Try OCR on each region
    for idx, (rx, ry, rx2, ry2) in enumerate(search_regions):
        crop = img[ry:ry2, rx:rx2]
        if crop.size == 0:
            continue

        if save_debug:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/search_roi_{idx:02d}_{rx}_{ry}.png", crop)

        # CRITICAL: Pass backend parameter to OCR
        txt = ocr_read_number(
            crop,
            debug_dir=debug_dir if save_debug else None,
            backend=ocr_backend,  # Pass the backend choice
        )

        if txt:
            parsed = parse_scale_text(txt)
            if parsed:
                val_nm, _ = parsed
                print(
                    f"✓ OCR Success at region {idx} ({rx},{ry}): '{txt}' → {val_nm} nm"
                )
                text_bbox = (rx, ry, rx2 - rx, ry2 - ry)
                return float(val_nm), text_bbox

    # Fallback to filename parsing
    val = _parse_from_filename(image_path)
    if val:
        print(f"✓ Parsed from filename: {val} nm")
        return val, None

    print("✗ OCR failed in all regions and filename has no scale info")
    return None, None


def detect_scale_bar_length(image_path: str):
    """
    Backward-compatible wrapper for legacy code.

    Returns
    -------
    width_px : int
        Scale bar width in pixels
    contour : None
        Legacy placeholder (no longer used, always None)
    """
    width_px, bbox, mask, _ = detect_scale_bar(
        image_path, save_debug=True, debug_dir="outputs/figures"
    )
    return width_px, None
