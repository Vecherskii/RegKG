import json
import os
import re
import tempfile

import fitz  # PyMuPDF

from pipeline.emit import process_pdf_to_jsonl

EL_ID_RE = re.compile(r"^pg\d{2}#el\d{4}$")
CH_ID_RE = re.compile(r"^[^#]+#pg=\d+#t=[a-z_]+#n=\d+$")

def _make_pdf_with_everything():
    doc = fitz.open()
    page = doc.new_page(width=595.28, height=841.89)
    x = 72

    # Heading
    y = 72
    page.insert_text((x, y), "Disclosure Requirements", fontsize=18)
    # Paragraph
    y += 28
    page.insert_text((x, y), "This paragraph explains the rules.", fontsize=12)
    # List item
    y += 20
    page.insert_text((x, y), "1) First requirement is documented.", fontsize=12)
    # Caption-like
    y += 40
    page.insert_text((x, y), "Figure 1: Sample chart", fontsize=10)

    # Image (1x1 PNG scaled)
    import base64
    _PNG_1x1_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    png_bytes = base64.b64decode(_PNG_1x1_B64)
    rect = fitz.Rect(100, 300, 180, 380)
    page.insert_image(rect, stream=png_bytes, keep_proportion=False)

    # Link
    link_rect = fitz.Rect(72, 72, 250, 90)
    page.insert_link({"from": link_rect, "kind": fitz.LINK_URI, "uri": "https://example.com"})

    tmpdir = tempfile.mkdtemp(prefix="emit_")
    pdf_path = os.path.join(tmpdir, "doc.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path

def test_emit_pages_and_chunks_ok():
    pdf = _make_pdf_with_everything()
    out_dir = tempfile.mkdtemp(prefix="emit_out_")
    pages_path, chunks_path = process_pdf_to_jsonl(pdf, out_dir, doc_id="DOC")

    assert os.path.exists(pages_path)
    assert os.path.exists(chunks_path)

    # Read back
    with open(pages_path, "r", encoding="utf-8") as f:
        pages_lines = [json.loads(line) for line in f]

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_lines = [json.loads(line) for line in f]

    # One page record
    assert len(pages_lines) == 1
    pg = pages_lines[0]
    assert pg["doc_id"] == "DOC"
    assert pg["page_num"] == 1
    assert pg["page_size"]["unit"] == "pt"
    assert isinstance(pg["parser_versions"]["pymupdf"], str)
    assert pg["parser_versions"]["docling"] is None

    # Elements and IDs
    assert len(pg["elements"]) >= 4
    for el in pg["elements"]:
        assert EL_ID_RE.match(el["element_id"])
        x0, y0, x1, y1 = el["bbox"]
        w, h = pg["page_size"]["w"], pg["page_size"]["h"]
        assert 0 <= x0 <= x1 <= w + 1e-3
        assert 0 <= y0 <= y1 <= h + 1e-3

    # Links present & shape ok
    assert isinstance(pg["links"], list)
    assert len(pg["links"]) >= 1
    lk = pg["links"][0]
    assert "https://" in lk["target"] or "http://" in lk["target"]

    # Chunks: schema + IDs + hashes
    assert len(chunks_lines) >= len(pg["elements"])  # text + image_area
    text_chunks = [c for c in chunks_lines if c["text"] is not None]
    image_chunks = [c for c in chunks_lines if c["chunk_type"] == "image_area"]

    assert len(text_chunks) >= 3
    assert len(image_chunks) >= 1

    for ch in chunks_lines:
        assert CH_ID_RE.match(ch["chunk_id"])
        assert ch["bbox_unit"] == "pt"
        assert ch["parser_version"] == "pymupdf_baseline_v1"
        if ch["text"] is not None:
            assert ch.get("text_hash", "").startswith("sha1-")
        else:
            assert "text_hash" not in ch
