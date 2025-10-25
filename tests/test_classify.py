import os
import tempfile

import fitz  # PyMuPDF

from pipeline.spans import extract_spans, spans_within_bounds
from pipeline.layout import group_spans_into_lines, group_lines_into_blocks
from pipeline.classify import classify_blocks


def _make_classify_pdf() -> str:
    doc = fitz.open()
    page = doc.new_page(width=595.28, height=841.89)
    x = 72

    # Heading (larger font)
    y = 72
    page.insert_text((x, y), "Disclosure Requirements", fontsize=18)

    # Paragraph (normal)
    y += 28
    page.insert_text((x, y), "This paragraph explains the rules.", fontsize=12)

    # List item
    y += 20
    page.insert_text((x, y), "1) First requirement is documented.", fontsize=12)

    # Caption (smaller, with Figure prefix)
    y += 40
    page.insert_text((x, y), "Figure 1: Sample chart", fontsize=10)

    tmpdir = tempfile.mkdtemp(prefix="classify_")
    pdf_path = os.path.join(tmpdir, "classify.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_block_classification_labels():
    pdf = _make_classify_pdf()
    doc = fitz.open(pdf)
    try:
        page = doc[0]
        spans = extract_spans(page, assume_pdf_bottom_left=False)
        assert spans_within_bounds(spans, (page.rect.width, page.rect.height))

        lines = group_spans_into_lines(spans)
        blocks = group_lines_into_blocks(lines)
        typed = classify_blocks(blocks, (page.rect.width, page.rect.height))

        # We expect at least 4 typed blocks
        types = [tb.type for tb in typed]

        assert "heading" in types, f"Labels: {types}"
        assert "paragraph" in types, f"Labels: {types}"
        assert "list_item" in types, f"Labels: {types}"
        assert "caption" in types, f"Labels: {types}"
    finally:
        doc.close()
