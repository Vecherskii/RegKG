from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF

from .spans import extract_spans, spans_within_bounds
from .layout import group_spans_into_lines, group_lines_into_blocks, Block
from .classify import classify_blocks, TypedBlock
from .images import extract_image_areas

BBox = Tuple[float, float, float, float]

# ----------------- Utilities -----------------

def _norm_ws(s: str) -> str:
    return " ".join(s.lower().split())

def _sha1_text(s: str) -> str:
    return "sha1-" + hashlib.sha1(_norm_ws(s).encode("utf-8")).hexdigest()

def _pymupdf_version() -> str:
    v = getattr(fitz, "__version__", None)
    if v:
        return v
    doc = getattr(fitz, "__doc__", "") or ""
    parts = doc.split()
    return parts[-1] if parts else "unknown"

def _bbox_order_key(b: BBox) -> Tuple[float, float, float, float]:
    return (float(b[1]), float(b[0]), float(b[3]), float(b[2]))

def _confidence_for_chunk_type(t: str) -> float:
    if t in ("heading", "paragraph", "list_item"):
        return 0.9
    if t == "caption":
        return 0.8
    if t in ("header", "footer"):
        return 0.75
    if t in ("image_area", "table_area", "formula_area"):
        return 0.7
    return 0.8

def _compute_doc_id(pdf_path: str) -> str:
    """
    Always compute a unique, deterministic doc_id from the file name and bytes.
    Format: <stem>_<sha1[:8]>
    """
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    h = hashlib.sha1()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"{stem}_{h.hexdigest()[:8]}"

# ----------------- Core Writer -----------------

@dataclass
class PageOutput:
    page_record: dict
    chunk_records: List[dict]

def process_page(doc_id: str, page: fitz.Page, page_num: int) -> PageOutput:
    """
    Build one pages.jsonl object and its corresponding chunks for this page.
    """
    pr = page.rect
    page_w, page_h = float(pr.width), float(pr.height)

    # 1) extract text / layout / types
    spans = extract_spans(page, assume_pdf_bottom_left=False)
    assert spans_within_bounds(spans, (page_w, page_h)), "Span outside page bounds"

    lines = group_spans_into_lines(spans)
    blocks: List[Block] = group_lines_into_blocks(lines)
    typed_blocks: List[TypedBlock] = classify_blocks(blocks, (page_w, page_h))

    # 2) images
    image_areas = extract_image_areas(page)

    # 3) links
    links_raw = page.get_links()
    links: List[dict] = []
    for lk in links_raw or []:
        r = lk.get("from") or lk.get("rect") or lk.get("bbox")
        if not r:
            continue
        bbox = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
        kind = lk.get("kind") or lk.get("type") or "uri"
        target = lk.get("uri") or lk.get("dest") or lk.get("file") or ""
        links.append({"from_bbox": bbox, "target": target, "kind": str(kind)})

    # 4) page-level metadata
    page_id = f"{doc_id}#pg={page_num}"
    parser_versions = {"pymupdf": _pymupdf_version(), "docling": None}

    # 5) elements (deterministic order)
    elements: List[dict] = []

    # text elements
    for tb in typed_blocks:
        elements.append({
            "element_id": "",
            "type": tb.type,
            "text": tb.text,
            "bbox": [float(tb.bbox[0]), float(tb.bbox[1]), float(tb.bbox[2]), float(tb.bbox[3])],
        })

    # image placeholders
    for ia in image_areas:
        el = {
            "element_id": "",
            "type": "image_area",
            "text": None,
            "bbox": [float(ia.bbox[0]), float(ia.bbox[1]), float(ia.bbox[2]), float(ia.bbox[3])],
        }
        if ia.xref is not None:
            el["payload"] = {"xref": int(ia.xref)}
        elements.append(el)

    elements.sort(key=lambda e: _bbox_order_key(tuple(e["bbox"])))

    for i, el in enumerate(elements, start=1):
        el["element_id"] = f"pg{page_num:02d}#el{i:04d}"

    page_record = {
        "page_id": page_id,
        "doc_id": doc_id,
        "page_num": page_num,
        "page_size": {"w": page_w, "h": page_h, "unit": "pt"},
        "parser_versions": parser_versions,
        "elements": elements,
        "links": links,
    }

    # 6) flattened chunks
    type_counters: Dict[str, int] = {}
    chunk_records: List[dict] = []
    for el in elements:
        ctype = el["type"]
        type_counters[ctype] = type_counters.get(ctype, 0) + 1
        idx = type_counters[ctype]

        text = el.get("text", None)
        rec: dict = {
            "chunk_id": f"{doc_id}#pg={page_num}#t={ctype}#n={idx}",
            "doc_id": doc_id,
            "page_num": page_num,
            "chunk_type": ctype,
            "text": text if ctype not in ("image_area", "table_area", "formula_area") else None,
            "bbox": el["bbox"],
            "bbox_unit": "pt",
            "page_size": {"w": page_w, "h": page_h, "unit": "pt"},
            "confidence": _confidence_for_chunk_type(ctype),
            "parser_version": "pymupdf_baseline_v1",
        }
        if rec["text"] is not None:
            rec["text_hash"] = _sha1_text(str(rec["text"]))
        chunk_records.append(rec)

    return PageOutput(page_record=page_record, chunk_records=chunk_records)

# ----------------- Top-level Document Runner -----------------

def process_pdf_to_jsonl(pdf_path: str, out_dir: str, doc_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Parse a PDF and write `pages.jsonl` and `chunks.jsonl` to `out_dir`.

    Always computes a unique doc_id = <stem>_<sha1[:8]> to avoid collisions.
    The `doc_id` parameter is ignored to enforce global uniqueness.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    computed_doc_id = _compute_doc_id(pdf_path)

    pages_path = os.path.join(out_dir, "pages.jsonl")
    chunks_path = os.path.join(out_dir, "chunks.jsonl")

    doc = fitz.open(pdf_path)
    try:
        with open(pages_path, "w", encoding="utf-8") as fp_pages, open(chunks_path, "w", encoding="utf-8") as fp_chunks:
            for i in range(len(doc)):
                page = doc[i]
                out = process_page(computed_doc_id, page, page_num=i + 1)
                fp_pages.write(json.dumps(out.page_record, ensure_ascii=False) + "\n")
                for ch in out.chunk_records:
                    fp_chunks.write(json.dumps(ch, ensure_ascii=False) + "\n")
    finally:
        doc.close()

    return pages_path, chunks_path
