import os
import tempfile

import fitz  # PyMuPDF

from pipeline.spans import extract_spans, spans_within_bounds
from pipeline.layout import group_spans_into_lines, group_lines_into_blocks


def _make_two_para_pdf() -> str:
    doc = fitz.open()
    page = doc.new_page(width=595.28, height=841.89)

    # Paragraph 1: force a line break with a hyphenated wrap
    x, y = 72, 100
    fontsize = 12
    para1_line1 = "Regulatory disclo-"
    para1_line2 = "sure requirements apply."
    page.insert_text((x, y), para1_line1, fontsize=fontsize)
    page.insert_text((x, y + 16), para1_line2, fontsize=fontsize)

    # Paragraph 2 with a bigger vertical gap
    para2_y = y + 16 + 30
    para2 = "Second paragraph starts here."
    page.insert_text((x, para2_y), para2, fontsize=fontsize)

    tmpdir = tempfile.mkdtemp(prefix="grouping_")
    pdf_path = os.path.join(tmpdir, "two_paras.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_grouping_lines_and_blocks():
    pdf = _make_two_para_pdf()
    doc = fitz.open(pdf)
    try:
        page = doc[0]
        spans = extract_spans(page, assume_pdf_bottom_left=False)  # <-- no flip
        assert spans_within_bounds(spans, (page.rect.width, page.rect.height))

        lines = group_spans_into_lines(spans)
        # Expect at least 3 lines (two for para1, one for para2)
        assert len(lines) >= 3

        blocks = group_lines_into_blocks(lines)
        # Expect 2 blocks (paragraphs)
        assert len(blocks) == 2

        # De-hyphenation check across line boundary of block 1
        b1_text = blocks[0].text.replace("\n", " ")
        assert "disclosure" in b1_text.lower(), f"Expected de-hyphenation, got: {b1_text}"

        # Bboxes within page
        w, h = float(page.rect.width), float(page.rect.height)
        for blk in blocks:
            x0, y0, x1, y1 = blk.bbox
            assert 0 <= x0 <= x1 <= w + 1e-3
            assert 0 <= y0 <= y1 <= h + 1e-3
    finally:
        doc.close()
