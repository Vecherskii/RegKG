from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Span:
    """
    A single text span extracted from a page.

    Attributes
    ----------
    text : str
        Raw text of the span (as reported by PyMuPDF).
    bbox : BBox
        (x0, y0, x1, y1) in points, origin at top-left, y increases downward.
    font : str
        Reported font name (may include style info).
    size : float
        Reported font size in points.
    flags : int
        Bit flags from PyMuPDF indicating style (bold/italic etc.).
    """
    text: str
    bbox: BBox
    font: str
    size: float
    flags: int


def pymupdf_version() -> str:
    """
    Return the PyMuPDF version string (e.g., '1.24.10').
    """
    return getattr(fitz, "__doc__", "").split()[-1] if hasattr(fitz, "__doc__") else getattr(fitz, "__version__", "unknown")


def _to_top_left_y_down(bbox: BBox, page_height: float, assume_pdf_bottom_left: bool) -> BBox:
    """
    Convert a rectangle to top-left origin, y-down coordinates if needed.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Input rectangle (x0, y0, x1, y1) as reported by PyMuPDF.
    page_height : float
        Page height in points.
    assume_pdf_bottom_left : bool
        If True, assume incoming coords are bottom-left origin (PDF space) and flip.
        If False, pass-through (already top-left origin).

    Returns
    -------
    tuple[float, float, float, float]
        Converted (x0, y0, x1, y1) with origin at top-left, y increasing downward.
    """
    x0, y0, x1, y1 = bbox
    if not assume_pdf_bottom_left:
        return (float(x0), float(y0), float(x1), float(y1))
    # Flip Y: PDF bottom-left -> top-left
    new_y0 = float(page_height - y1)
    new_y1 = float(page_height - y0)
    return (float(x0), new_y0, float(x1), new_y1)


def extract_spans(page: fitz.Page, assume_pdf_bottom_left: bool = False) -> List[Span]:
    """
    Extract span-level text from a PyMuPDF page and normalize coordinates.

    IMPORTANT
    ---------
    Modern PyMuPDF returns text geometry in top-left / y-down already.
    Therefore, the default is `assume_pdf_bottom_left=False` to avoid double flipping.

    Parameters
    ----------
    page : fitz.Page
        The page object to extract from.
    assume_pdf_bottom_left : bool, default False
        Set True only if you're certain the input rectangles are bottom-left origin.

    Returns
    -------
    list[Span]
        One Span per text span encountered on the page.

    Raises
    ------
    ValueError
        If page has no rect or page size is invalid.
    """
    pr = page.rect
    page_w, page_h = float(pr.width), float(pr.height)
    if page_w <= 0 or page_h <= 0:
        raise ValueError("Invalid page size reported by PyMuPDF")

    raw = page.get_text("dict")
    out: List[Span] = []

    for block in raw.get("blocks", []):
        if block.get("type") == 1:  # images handled later
            continue
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                text: str = s.get("text", "")
                bbox_raw = tuple(s.get("bbox", (0, 0, 0, 0)))
                font: str = s.get("font", "")
                size: float = float(s.get("size", 0.0))
                flags: int = int(s.get("flags", 0))

                bbox = _to_top_left_y_down(
                    bbox=bbox_raw, page_height=page_h, assume_pdf_bottom_left=assume_pdf_bottom_left
                )
                out.append(Span(text=text, bbox=bbox, font=font, size=size, flags=flags))

    return out


def spans_within_bounds(spans: Iterable[Span], page_size: Tuple[float, float]) -> bool:
    """
    Check that all span bboxes lie within [0, w] x [0, h].

    Parameters
    ----------
    spans : Iterable[Span]
        Spans to validate.
    page_size : (float, float)
        (width, height) in points.

    Returns
    -------
    bool
        True if all bboxes are within page bounds (allowing a tiny epsilon).
    """
    w, h = float(page_size[0]), float(page_size[1])
    eps = 1e-3
    for sp in spans:
        x0, y0, x1, y1 = sp.bbox
        if x0 < -eps or y0 < -eps or x1 > w + eps or y1 > h + eps or x0 > x1 + eps or y0 > y1 + eps:
            return False
    return True
