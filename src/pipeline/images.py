from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from .spans import BBox


@dataclass(frozen=True)
class ImageArea:
    """
    Detected image area on a page.

    Attributes
    ----------
    bbox : BBox
        (x0, y0, x1, y1) in points, origin at top-left, y increasing downward.
    xref : Optional[int]
        The image XREF number if PyMuPDF exposed it; None otherwise.
    """
    bbox: BBox
    xref: Optional[int] = None


def extract_image_areas(page: fitz.Page) -> List[ImageArea]:
    """
    Extract image areas from a page using PyMuPDF's 'dict' text map.

    Parameters
    ----------
    page : fitz.Page
        The page to analyze.

    Returns
    -------
    list[ImageArea]
        One ImageArea per image block detected, preserving top-left / y-down coords.

    Notes
    -----
    PyMuPDF's page.get_text("dict") includes blocks with "type" == 1 for images.
    These typically include:
        - "bbox": [x0, y0, x1, y1] in page coordinates,
        - "image": <xref int>  (not always present, but common).
    """
    raw = page.get_text("dict")
    out: List[ImageArea] = []

    for block in raw.get("blocks", []):
        if int(block.get("type", -1)) != 1:
            continue
        bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        # PyMuPDF dict is already top-left origin, y-down
        xref = block.get("image", None)
        try:
            xref_int = int(xref) if xref is not None else None
        except Exception:
            xref_int = None
        out.append(ImageArea(bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])), xref=xref_int))

    return out
