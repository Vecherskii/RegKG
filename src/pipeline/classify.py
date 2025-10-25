from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import re

from .layout import Block, Line
from .spans import BBox

# Tunables
DELTA_LARGE = 2.0
DELTA_SMALL = 1.5
HEADING_RATIO = 1.25
HEADING_MAX_CHARS = 80
HEADER_FOOTER_BAND = 0.08
HEADER_FOOTER_SIZE_SLACK = 0.5

BOLD_HINTS = ("bold", "black", "semibold", "demibold", "heavy")

LIST_PATTERNS = [
    r"^\s*[\u2022\-\–\—\*]\s+",
    r"^\s*\(?\d+[\.\)\]]\s+",
    r"^\s*\(?[a-zA-Z][\.\)\]]\s+",
]
LIST_RE = [re.compile(p) for p in LIST_PATTERNS]
CAPTION_PREFIX_RE = re.compile(r"^\s*(figure|fig\.|table|tab\.)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TypedBlock:
    type: str
    text: str
    bbox: BBox
    avg_size: float
    lines: Tuple[Line, ...]


def _median(values: List[float]) -> float:
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    mid = n // 2
    return (v[mid - 1] + v[mid]) / 2 if n % 2 == 0 else v[mid]


def _looks_bold(lines: Iterable[Line]) -> bool:
    for ln in lines:
        for sp in ln.spans:
            f = sp.font.lower()
            if any(h in f for h in BOLD_HINTS):
                return True
    return False


def _is_list_text(text: str) -> bool:
    for r in LIST_RE:
        if r.search(text):
            return True
    return False


def classify_blocks(blocks: List[Block], page_size: Tuple[float, float]) -> List[TypedBlock]:
    if not blocks:
        return []

    line_sizes: List[float] = [ln.avg_size for b in blocks for ln in b.lines if ln.avg_size > 0]
    median_size = _median(line_sizes) or 10.0

    w, h = float(page_size[0]), float(page_size[1])
    top_band = HEADER_FOOTER_BAND * h
    bot_band = h - top_band

    typed: List[TypedBlock] = []
    for b in blocks:
        text = (b.text or "").strip()
        x0, y0, x1, y1 = b.bbox
        size = b.avg_size

        # Header / Footer
        if y1 <= top_band and size <= (median_size + HEADER_FOOTER_SIZE_SLACK):
            typed.append(TypedBlock("header", text, b.bbox, size, b.lines)); continue
        if y0 >= bot_band and size <= (median_size + HEADER_FOOTER_SIZE_SLACK):
            typed.append(TypedBlock("footer", text, b.bbox, size, b.lines)); continue

        # Prefer list if the block's FIRST non-empty line starts like a list
        first_line_text = next((ln.text for ln in b.lines if (ln.text or "").strip()), "")
        if _is_list_text(first_line_text):
            typed.append(TypedBlock("list_item", text, b.bbox, size, b.lines)); continue

        # Caption
        if CAPTION_PREFIX_RE.search(text) or size <= (median_size - DELTA_SMALL):
            typed.append(TypedBlock("caption", text, b.bbox, size, b.lines)); continue

        # Heading
        if (size >= median_size + DELTA_LARGE) or (size >= HEADING_RATIO * median_size and len(text) <= HEADING_MAX_CHARS) or _looks_bold(b.lines):
            typed.append(TypedBlock("heading", text, b.bbox, size, b.lines)); continue

        # Default
        typed.append(TypedBlock("paragraph", text, b.bbox, size, b.lines))

    return typed
