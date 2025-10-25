import os
import tempfile

import fitz  # PyMuPDF

from pipeline.spans import extract_spans, spans_within_bounds


def _make_demo_pdf(text_lines=("Hello", "World"), page_size=(595.28, 841.89)) -> str:
    """Create a small single-page PDF with two lines and return its path."""
    doc = fitz.open()
    page = doc.new_page(width=page_size[0], height=page_size[1])

    x, y = 72, 72
    for i, line in enumerate(text_lines):
        page.insert_text((x, y + i * 18), line, fontsize=12)

    tmpdir = tempfile.mkdtemp(prefix="pymu_spans_")
    pdf_path = os.path.join(tmpdir, "demo.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extract_spans_and_bounds():
    pdf = _make_demo_pdf()
    doc = fitz.open(pdf)
    try:
        page = doc[0]
        # IMPORTANT: PyMuPDF is already top-left / y-down; do NOT flip.
        spans = extract_spans(page, assume_pdf_bottom_left=False)
        assert len(spans) >= 2

        w, h = float(page.rect.width), float(page.rect.height)
        assert spans_within_bounds(spans, (w, h)), "Some span bbox is out of page bounds"
    finally:
        doc.close()
