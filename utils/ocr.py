"""
Lightweight OCR utilities for reading scale bar text like "50 nm" or "0.2 µm".

Design goals:
-------------
* Be optional: if OCR backends aren't installed yet, the pipeline still runs
  (you can pass --scale from CLI). The detector never crashes.
* Support either pytesseract (needs OS tesseract-ocr) or EasyOCR (pure pip).
* Normalize µm/um to nm so downstream logic only deals with nanometers.
"""

import re
from typing import Optional, Tuple

# Keep a small registry of available OCR backends
_BACKENDS = []

try:
    import pytesseract  # type: ignore
    _BACKENDS.append("tesseract")
except Exception:
    pass

try:
    import easyocr  # type: ignore
    _BACKENDS.append("easyocr")
except Exception:
    pass


def parse_scale_text(text: str) -> Optional[Tuple[float, str]]:
    """
    Parse OCR'd text and extract a numeric value with a unit.

    Accepted patterns (examples):
    -----------------------------
    - "50 nm", "100nm", "200 nm."
    - "0.2 µm", "0.20um"

    Returns
    -------
    (value_nm, "nm") if the unit is recognized and normalized; otherwise None.
    """
    if not text:
        return None

    # Normalize common variants that OCR might produce:
    t = text.strip()
    t = t.replace("μ", "µ")   # greek mu → micro sign
    t = t.replace("u", "µ")   # plain 'u' often appears instead of 'µ'
    t = t.replace(",", ".")   # support comma decimal separators

    # Capture a number (int or float) followed by nm / µm / um (case-insensitive)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(nm|µm|um)\b", t, flags=re.IGNORECASE)
    if not m:
        return None

    val = float(m.group(1))
    unit = m.group(2).lower()

    # Normalize to nm so the pipeline uses a single unit
    if unit == "nm":
        return (val, "nm")
    if unit in ("µm", "um"):
        return (val * 1000.0, "nm")

    return None


def ocr_read_number(image, lang_hint: str = "eng") -> Optional[str]:
    """
    Run OCR using the first available backend and return raw text if any.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed (ideally thresholded) grayscale crop.
    lang_hint : str, optional (default="eng")
        Language hint for OCR (kept simple here).

    Returns
    -------
    text : str or None
        The raw OCR string, or None if no backend is available or nothing is read.
    """
    for backend in _BACKENDS:
        try:
            if backend == "tesseract":
                from pytesseract import image_to_string  # type: ignore
                # Whitelist numerals + nm/µm letters to bias OCR towards scale strings
                custom = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789nmµuUM."
                txt = image_to_string(image, config=custom, lang=lang_hint)
                if txt and txt.strip():
                    return txt

            elif backend == "easyocr":
                import easyocr  # type: ignore
                reader = easyocr.Reader([lang_hint], gpu=False)
                res = reader.readtext(image)
                if res:
                    # Concatenate recognized chunks into one string
                    return " ".join(r[1] for r in res if r and len(r) >= 2)
        except Exception:
            # If a backend errors (e.g., missing binaries), silently try the next one.
            continue

    return None
