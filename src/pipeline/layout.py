from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Sequence
import re

from .spans import Span, BBox

# ---- Tunable constants (baseline heuristics) ----
VERT_OVERLAP_MIN = 0.60          # min vertical overlap ratio for same-line
BASELINE_TOL_PT = 1.5            # absolute baseline diff (pt) allowed
BASELINE_TOL_SCALE = 0.25        # * median span size (pt)
BLOCK_FONT_DIFF_MAX = 1.5        # max avg font size difference (pt)
ABS_BASELINE_GAP_FLOOR_PT = 24.0 # absolute min baseline jump that forces a new block
JOIN_HYPHENS = True              # enable boundary fusion

# Detect lines that likely start list items (•, -, 1), 1., (a), a., etc.)
_LIST_PATTERNS = [
    r"^\s*[\u2022\-\–\—\*]\s+",          # bullets: • - – — *
    r"^\s*\(?\d+[\.\)\]]\s+",            # 1.  1)  1]
    r"^\s*\(?[a-zA-Z][\.\)\]]\s+",       # a.  b)  A)  A.
]
LIST_START_RE = re.compile("|".join(_LIST_PATTERNS))


@dataclass(frozen=True)
class Line:
    """A single text line made of one or more spans."""
    text: str
    bbox: BBox
    spans: Tuple[Span, ...]
    avg_size: float
    baseline_y: float


@dataclass(frozen=True)
class Block:
    """A paragraph-like block consisting of one or more lines."""
    text: str
    bbox: BBox
    lines: Tuple[Line, ...]
    avg_size: float


def _union_bbox(rects: Sequence[BBox]) -> BBox:
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[2] for r in rects)
    y1 = max(r[3] for r in rects)
    return (x0, y0, x1, y1)


def _vertical_overlap(a: BBox, b: BBox) -> float:
    """Return vertical overlap ratio of two rects relative to the smaller height."""
    ay0, ay1 = a[1], a[3]
    by0, by1 = b[1], b[3]
    inter = max(0.0, min(ay1, by1) - max(ay0, by0))
    h_min = max(1e-6, min(ay1 - ay0, by1 - by0))
    return inter / h_min


def _median(values: Sequence[float]) -> float:
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    mid = n // 2
    return (v[mid - 1] + v[mid]) / 2 if n % 2 == 0 else v[mid]


def _concat_spans_text(spans: Sequence[Span]) -> str:
    # PyMuPDF spans already include spaces when present
    return "".join(s.text for s in spans)


def _leading_alpha(fragment: str) -> Tuple[int, str]:
    """
    Return (start_index, leading_alpha_token) from `fragment`, skipping any
    non-letters before the first alphabetic character. If none, returns (-1, "").
    """
    n = len(fragment)
    i = 0
    while i < n and not fragment[i].isalpha():
        i += 1
    if i >= n:
        return -1, ""
    j = i
    while j < n and fragment[j].isalpha():
        j += 1
    return i, fragment[i:j]


def _trailing_alpha(fragment: str) -> str:
    """Return the trailing alphabetic token of `fragment` (possibly empty)."""
    k = len(fragment) - 1
    while k >= 0 and not fragment[k].isalpha():
        k -= 1
    if k < 0:
        return ""
    j = k
    while j >= 0 and fragment[j].isalpha():
        j -= 1
    return fragment[j + 1 : k + 1]


def _maybe_dehyphenate(prev: str, curr: str) -> Tuple[str, str]:
    """
    Join a line boundary conservatively, handling explicit hyphens and soft wraps.

    Unified behavior:
      - Compute the next line's leading alphabetic token (skipping spaces).
      - If prev ends with '-', '–', or '—' **and** there is a leading token:
            drop the hyphen and fuse that token to prev; remove it from curr.
      - Else if prev ends with letters **and** next has a leading token:
            fuse that token to prev; remove it from curr.
    """
    if not JOIN_HYPHENS:
        return prev, curr

    prev_stripped = prev.rstrip()
    curr_stripped = curr.lstrip()

    i, curr_head = _leading_alpha(curr_stripped)
    prev_tail = _trailing_alpha(prev_stripped)

    if not curr_head:
        return prev, curr

    if prev_stripped.endswith(("-", "–", "—")):
        fused_prev = prev_stripped[:-1] + curr_head
        remainder_curr = curr_stripped[:i] + curr_stripped[i + len(curr_head) :]
        return fused_prev, remainder_curr

    if prev_tail:
        fused_prev = prev_stripped + curr_head
        remainder_curr = curr_stripped[:i] + curr_stripped[i + len(curr_head) :]
        return fused_prev, remainder_curr

    return prev, curr


def group_spans_into_lines(spans: Iterable[Span]) -> List[Line]:
    """
    Group normalized spans into reading-order lines.

    Two spans are on the same line if vertical overlap >= VERT_OVERLAP_MIN,
    or their baselines differ less than a tolerance derived from font sizes.
    """
    spans_sorted = sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0]))
    if not spans_sorted:
        return []

    median_size = _median([s.size for s in spans_sorted if s.size > 0]) or 10.0
    baseline_tol = max(BASELINE_TOL_PT, BASELINE_TOL_SCALE * median_size)

    lines: List[List[Span]] = []
    for sp in spans_sorted:
        placed = False
        for line_spans in lines[::-1]:  # try most recent line first
            last = line_spans[-1]
            ov = _vertical_overlap(last.bbox, sp.bbox)
            baseline_diff = abs(last.bbox[3] - sp.bbox[3])  # bottom y as baseline proxy
            if ov >= VERT_OVERLAP_MIN or baseline_diff <= baseline_tol:
                line_spans.append(sp)
                placed = True
                break
        if not placed:
            lines.append([sp])

    out: List[Line] = []
    for line_spans in lines:
        line_spans.sort(key=lambda s: (s.bbox[0], s.bbox[1]))
        text = _concat_spans_text(line_spans)
        bbox = _union_bbox([s.bbox for s in line_spans])
        avg_size = sum(s.size for s in line_spans) / max(1, len(line_spans))
        baseline_y = max(s.bbox[3] for s in line_spans)
        out.append(Line(text=text, bbox=bbox, spans=tuple(line_spans), avg_size=avg_size, baseline_y=baseline_y))

    out.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return out


def group_lines_into_blocks(lines: Iterable[Line]) -> List[Block]:
    """
    Group lines into paragraph-like blocks using baseline jump, list-starts, and font-size similarity.

    New block if ANY of:
      - current line looks like a list start (LIST_START_RE),
      - baseline_gap(prev -> curr) > max(1.3 * max(prev_h, curr_h), ABS_BASELINE_GAP_FLOOR_PT), OR
      - avg font sizes differ by more than BLOCK_FONT_DIFF_MAX.
    """
    lns = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    if not lns:
        return []

    blocks: List[List[Line]] = [[lns[0]]]
    for prev, curr in zip(lns, lns[1:]):
        # 1) hard split on list starters to avoid merging list lines into a paragraph
        if LIST_START_RE.search(curr.text or ""):
            blocks.append([curr])
            continue

        prev_h = max(1.0, prev.bbox[3] - prev.bbox[1])
        curr_h = max(1.0, curr.bbox[3] - curr.bbox[1])

        baseline_gap = curr.baseline_y - prev.baseline_y
        gap_threshold = max(1.3 * max(prev_h, curr_h), ABS_BASELINE_GAP_FLOOR_PT)
        font_band_ok = abs(prev.avg_size - curr.avg_size) <= BLOCK_FONT_DIFF_MAX

        if (baseline_gap > gap_threshold) or (not font_band_ok):
            blocks.append([curr])
        else:
            blocks[-1].append(curr)

    result: List[Block] = []
    for lines_in_block in blocks:
        texts: List[str] = []
        for idx, ln in enumerate(lines_in_block):
            t = ln.text
            if idx > 0:
                prev_txt = texts[-1]
                prev_new, curr_new = _maybe_dehyphenate(prev_txt.rstrip(), t.lstrip())
                texts[-1] = prev_new
                t = curr_new
            texts.append(t)
        block_text = "\n".join(texts)

        bbox = _union_bbox([ln.bbox for ln in lines_in_block])
        avg_size = sum(ln.avg_size for ln in lines_in_block) / max(1, len(lines_in_block))
        result.append(Block(text=block_text, bbox=bbox, lines=tuple(lines_in_block), avg_size=avg_size))

    result.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return result
