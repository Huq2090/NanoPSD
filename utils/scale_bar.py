# Import OpenCV for image processing and NumPy for numerical operations
import cv2
import numpy as np
from typing import Tuple, Optional

# Try to import our OCR helpers. If OCR backends aren't installed yet,
# we provide safe stubs so the detector still works (just without auto-cal).
try:
    from .ocr import ocr_read_number, parse_scale_text
except Exception:

    def ocr_read_number(image, lang_hint: str = "eng"):
        """Fallback OCR stub that returns None when OCR backends are unavailable."""
        return None

    def parse_scale_text(text: str):
        """Fallback parser stub that returns None when OCR backends are unavailable."""
        return None


def _bottom_band_roi(
    image: np.ndarray, frac: float = 0.25
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a bottom band Region Of Interest (ROI) where scale bars typically live.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image array.
    frac : float, optional (default=0.25)
        Fraction of image height to keep in the ROI from the bottom.

    Returns
    -------
    roi : np.ndarray
        Cropped grayscale ROI (bottom band of the image).
    roi_box : (x, y, w, h)
        ROI coordinates relative to the full image. Here x=0, y=top of bottom band.
    """
    h, w = image.shape[:2]
    # Clamp fraction to avoid pathological values
    frac = max(0.05, min(frac, 0.5))
    y0 = int(h * (1.0 - frac))
    roi = image[y0:h, 0:w]
    return roi, (0, y0, w, h - y0)


def _threshold_and_candidates(roi_gray: np.ndarray):
    """
    Produce contour candidates from the ROI by trying both polarities.

    Why both polarities?
    --------------------
    Some micrographs render the scale bar as dark on light (black bar), others as
    light on dark (white bar). Trying the inverted image increases robustness.

    Returns
    -------
    list of tuples: (binary_mask, contours)
    """
    results = []
    for invert in (False, True):
        g = roi_gray if not invert else (255 - roi_gray)

        # Otsu threshold → high-contrast binary mask of potential bar regions
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Light morphological opening removes speckle noise while preserving long bars
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        # External contours are sufficient: the scale bar is an outermost object
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results.append((bw, contours))
    return results


def _score_bar_candidate(
    w: int,
    h: int,
    y_center: float,
    roi_h: int,
    solidity: float,
    extent: float,
    dist_edge: float,
) -> float:
    """
    Score a candidate blob to decide how "bar-like" it is.

    Intuition:
    ----------
    - Aspect ratio: scale bars are long & thin → high w/h
    - Solidity / Extent: bars are usually rectangular & solid
    - Vertical position: typically near the bottom edge of the ROI
    - Distance to corner/edges: bars often hug corners/bottom margins

    Returns
    -------
    score : float
        Higher is better. The absolute value doesn't matter; it's used for ranking.
    """
    aspect = w / max(h, 1)  # Avoid division by zero
    # Ideal vertical placement ~90% down within the ROI band
    near_bottom = 1.0 - abs((y_center / max(roi_h, 1)) - 0.9)

    score = (
        0.45
        * min(aspect / 12.0, 1.0)  # Reward very long-thin shapes; saturate at AR~12
        + 0.20 * min(solidity, 1.0)  # Reward solid, convex shapes
        + 0.15
        * min(extent / 0.85, 1.0)  # Reward rectangle-like fill of its bounding box
        + 0.10 * near_bottom  # Prefer lower positions in the ROI
        + 0.10
        * (
            1.0 - min(dist_edge / 30.0, 1.0)
        )  # Prefer near edges/corners (small distance)
    )
    return float(score)


def _mask_from_bbox(
    shape: Tuple[int, int], bbox: Tuple[int, int, int, int], pad: int = 2
) -> np.ndarray:
    """
    Create a binary mask covering the bar's bounding box (with a small padding).

    This is later used to ensure the bar region is excluded from particle analysis.

    Parameters
    ----------
    shape : (H, W)
        Shape of the full image.
    bbox : (x, y, w, h)
        Bar bounding box in full-image coordinates.
    pad : int, optional (default=2)
        Padding in pixels to slightly expand the masked region.

    Returns
    -------
    mask : np.ndarray (uint8)
        0 everywhere; 255 in the bar region.
    """
    H, W = shape[:2]
    x, y, w, h = bbox
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, W)
    y1 = min(y + h + pad, H)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def detect_scale_bar(
    image_path: str,
    save_debug: bool = True,
    debug_dir: str = "outputs/figures",
) -> Tuple[int, Tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """
    Hybrid, geometry-first scale bar detector.

    Strategy (robust across different microscopes/themes):
    -----------------------------------------------------
    1) Crop a bottom-band ROI where bars typically live.
    2) Threshold both polarities to get candidate blobs.
    3) Evaluate each contour using geometric cues (aspect, solidity, extent),
       proximity to bottom/corners, and keep the best candidate.
    4) If no candidates survive, fall back to Hough line detection for hairline bars.
    5) Return the bar width in pixels + its bbox + a mask for exclusion.

    Parameters
    ----------
    image_path : str
        Path to the input image (grayscale or color; read as grayscale).
    save_debug : bool, optional (default=True)
        If True, saves helpful debug overlays to `debug_dir`.
    debug_dir : str, optional (default="outputs/figures")
        Folder to write debug images.

    Returns
    -------
    width_px : int
        Detected bar length in pixels (width of the bounding box).
    bar_bbox : (x, y, w, h)
        Bounding box of the detected bar in full-image coordinates.
    bar_mask : np.ndarray (uint8)
        Binary mask of the bar region (255 inside bar box, else 0).
    roi_vis : np.ndarray (BGR)
        Visualization of the ROI with the winning candidate drawn (for debugging).

    Raises
    ------
    ValueError
        If no scale bar candidate is found.
    """
    # Step 0: Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Step 1: Focus search on the bottom band ROI to reduce false positives
    roi, (rx, ry, rw, rh) = _bottom_band_roi(img, frac=0.25)

    # Step 2: Collect & score candidates from both polarities
    best = None  # Will hold: (score, width_px, bbox_full, bar_mask, roi_vis)
    for bw, contours in _threshold_and_candidates(roi):
        for cnt in contours:
            if len(cnt) < 5:
                # Very tiny or degenerate contour; skip
                continue

            # Compute tight bounding box in the ROI coordinates
            x, y, cw, ch = cv2.boundingRect(cnt)

            # Quick gates to avoid expensive scoring on impossible candidates
            if ch < 1 or cw < 10:
                continue

            aspect = cw / max(ch, 1)
            if aspect < 5.0:
                # The scale bar should be long & thin; ignore stubbier shapes
                continue

            # Shape descriptors that favor rectangles / solid bars
            area = cv2.contourArea(cnt)
            rect_area = cw * ch
            extent = area / max(rect_area, 1)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)

            # Prefer bottom corners / edges
            dist_edge = min(x, y, rw - (x + cw), rh - (y + ch))

            # Score the candidate using a weighted combination of cues
            score = _score_bar_candidate(
                w=cw,
                h=ch,
                y_center=y + ch / 2.0,
                roi_h=rh,
                solidity=solidity,
                extent=extent,
                dist_edge=dist_edge,
            )

            # Convert bbox to full-image coordinates and build a mask now
            bbox_full = (x + rx, y + ry, cw, ch)
            bar_mask = _mask_from_bbox(img.shape, bbox_full, pad=2)

            # Keep the best-scoring candidate; tie-break on width
            if (
                (best is None)
                or (score > best[0])
                or (score == best[0] and cw > best[1])
            ):
                roi_vis = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
                cv2.rectangle(roi_vis, (x, y), (x + cw, y + ch), (0, 255, 255), 2)
                best = (score, cw, bbox_full, bar_mask, roi_vis)

    # Step 3: Fallback to Hough lines for thin hairline bars (if no candidates were found)
    if best is None:
        # Edge map → Hough P (probabilistic) to find long, near-horizontal lines
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=60,
            minLineLength=max(roi.shape[1] // 8, 10),
            maxLineGap=10,
        )
        if lines is not None and len(lines) > 0:
            best_len = 0
            best_bbox = None
            for x1, y1, x2, y2 in lines[:, 0, :]:
                length = int(np.hypot(x2 - x1, y2 - y1))
                # near-horizontal tolerance: |dy| <= 3 px
                if length > best_len and abs(y2 - y1) <= 3:
                    best_len = length
                    x = min(x1, x2)
                    y = min(y1, y2)
                    # Approximate a thin bar height of 3 pixels in the ROI
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
                # Save debug overlays if requested
                if save_debug:
                    import os

                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(f"{debug_dir}/scale_candidates_hough.png", roi_vis)
                return width_px, best_bbox, bar_mask, roi_vis

        # If we get here, no fallback line either → fail
        raise ValueError("Scale bar not detected.")

    # Step 4: Prepare outputs for the best geometric candidate
    _, width_px, bbox_full, bar_mask, roi_vis = best

    # Optional: Save a full-image visualization of the chosen bar
    if save_debug:
        import os

        os.makedirs(debug_dir, exist_ok=True)
        dbg = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        x, y, w, h = bbox_full
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imwrite(f"{debug_dir}/scale_bar_final.png", dbg)
        cv2.imwrite(f"{debug_dir}/scale_candidates.png", roi_vis)

    # Step 5: Return bar width (px), bbox, mask, and the ROI visualization
    return int(width_px), bbox_full, bar_mask, roi_vis


def detect_scale_label(
    image_path: str,
    bar_bbox: Tuple[int, int, int, int],
    save_debug: bool = True,
    debug_dir: str = "outputs/figures",
) -> Optional[float]:
    """
    OCR-read the printed scale text near the detected bar and convert to nanometers.

    Typical formats recognized:
    ---------------------------
    - "50 nm", "100nm", "200 nm."
    - "0.2 µm", "0.20um"   → automatically converted to nm

    Parameters
    ----------
    image_path : str
        Path to the input image.
    bar_bbox : (x, y, w, h)
        Bounding box of the scale bar in full-image coordinates.
    save_debug : bool, optional (default=True)
        If True, saves the OCR ROIs used for recognition.
    debug_dir : str, optional (default="outputs/figures")
        Folder to write debug crops for review.

    Returns
    -------
    value_nm : float or None
        Physical value in nanometers parsed from text; None if OCR fails.
    """
    import os, re

    # Step 0: Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Unpack bar box and get image dimensions
    x, y, w, h = bar_bbox
    H, W = img.shape[:2]

    # ---- helper: filename fallback like '..._50nm.png', '...-0.2um.tif' ----
    def _parse_from_filename(path: str) -> Optional[float]:
        name = os.path.basename(path)
        name = name.replace("μ", "µ").lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*nm\b", name)
        if m:
            return float(m.group(1))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(µm|um)\b", name)
        if m:
            return float(m.group(1)) * 1000.0
        return None

    # ---- helper: preprocess for OCR (with options) ----
    def _prep(im: np.ndarray, invert: bool, dilate: bool) -> np.ndarray:
        if invert:
            im = 255 - im
        im = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        im = cv2.GaussianBlur(im, (3, 3), 0)
        im = cv2.equalizeHist(im)
        # Otsu threshold
        _, th = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if dilate:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            th = cv2.dilate(th, k, iterations=1)
        return th

    # ---- candidate ROIs near the bar: right, left, above ----
    near_pads = [
        (w + 5, -2 * h, min(W, x + w + W // 3) - (x + w + 5), 3 * h),  # right
        (-W // 3, -2 * h, w - 5 + W // 3, 3 * h),  # left
        (-W // 3, -int(3 * h), W // 3 + w, int(4 * h)),  # above
    ]
    near_rois = []
    for dx, dy, ww, hh in near_pads:
        rx = max(0, x + dx)
        ry = max(0, y + dy)
        rx2 = min(W, rx + int(ww))
        ry2 = min(H, ry + int(hh))
        if (rx2 - rx) > 8 and (ry2 - ry) > 8:
            near_rois.append((rx, ry, rx2, ry2))

    # ---- helper: attempt OCR on a list of ROIs with multiple preprocess combos ----
    def _try_ocr_on_rois(rois):
        for rx, ry, rx2, ry2 in rois:
            crop = img[ry:ry2, rx:rx2]
            # Try normal/inverted × (no-dilate/dilate)
            for invert in (False, True):
                for dilate in (False, True):
                    th = _prep(crop, invert=invert, dilate=dilate)
                    txt = ocr_read_number(th)
                    if save_debug:
                        os.makedirs(debug_dir, exist_ok=True)
                        tag = f"inv{int(invert)}_dil{int(dilate)}"
                        cv2.imwrite(f"{debug_dir}/ocr_roi_{rx}_{ry}_{tag}.png", th)
                    if txt:
                        parsed = parse_scale_text(txt)
                        if parsed:
                            val_nm, _ = parsed
                            return float(val_nm)
        return None

    # 1) Try near-bar ROIs first
    val = _try_ocr_on_rois(near_rois)
    if val is not None:
        return val

    # 2) Widen search to a bottom strip (entire width, ~last 25% of height)
    by0 = int(H * 0.75)
    # Split into 3 vertical chunks (left/center/right) to keep ROIs manageable
    third = max(W // 3, 1)
    wide_rois = [
        (0, by0, third, H),
        (third, by0, min(2 * third, W), H),
        (min(2 * third, W), by0, W, H),
    ]
    val = _try_ocr_on_rois(wide_rois)
    if val is not None:
        return val

    # 3) Final fallback: filename pattern like '*_50nm.png'
    return _parse_from_filename(image_path)


def detect_scale_bar_length(image_path: str):
    """
    Backward-compatible wrapper for older pipeline code.

    Returns
    -------
    (max_width_px, scale_bar_contour)
        `scale_bar_contour` is no longer needed downstream; we return None.
    """
    width_px, bbox, mask, _ = detect_scale_bar(
        image_path, save_debug=True, debug_dir="outputs/figures"
    )
    return width_px, None
