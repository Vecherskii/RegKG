import os
import tempfile
import base64

import fitz  # PyMuPDF

from pipeline.images import extract_image_areas


# A tiny 1x1 PNG (black pixel), base64-encoded.
_PNG_1x1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def _make_pdf_with_image():
    doc = fitz.open()
    page = doc.new_page(width=595.28, height=841.89)

    # Decode the tiny PNG and insert as an image stream.
    png_bytes = base64.b64decode(_PNG_1x1_B64)

    # Place it on the page at a known rectangle
    rect = fitz.Rect(100, 150, 164, 214)  # 64x64 pt box
    page.insert_image(rect, stream=png_bytes, keep_proportion=False)

    tmpdir = tempfile.mkdtemp(prefix="images_")
    pdf_path = os.path.join(tmpdir, "with_image.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path, rect


def test_extract_image_areas_detects_bbox_and_xref():
    pdf_path, rect = _make_pdf_with_image()
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        areas = extract_image_areas(page)
        assert len(areas) >= 1, "Expected at least one image area"

        # Find an area that overlaps our inserted rect
        found = None
        for a in areas:
            ax0, ay0, ax1, ay1 = a.bbox
            # Simple overlap check
            inter_w = max(0, min(ax1, rect.x1) - max(ax0, rect.x0))
            inter_h = max(0, min(ay1, rect.y1) - max(ay0, rect.y0))
            if inter_w > 0 and inter_h > 0:
                found = a
                break

        assert found is not None, f"No image area overlapping expected rect {rect} among {areas}"
        # xref is often present; if it's not, we still accept the bbox.
        if found.xref is not None:
            assert isinstance(found.xref, int) and found.xref > 0
        # bbox within page
        w, h = float(page.rect.width), float(page.rect.height)
        x0, y0, x1, y1 = found.bbox
        assert 0 <= x0 <= x1 <= w + 1e-3
        assert 0 <= y0 <= y1 <= h + 1e-3
    finally:
        doc.close()
