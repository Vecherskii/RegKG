# src/main.py
from __future__ import annotations

import argparse
import os
from .pipeline.emit import process_pdf_to_jsonl, _compute_doc_id  # <-- relative import

def main():
    ap = argparse.ArgumentParser(description="PyMuPDF baseline PDF parser -> pages.jsonl / chunks.jsonl")
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    computed_id = _compute_doc_id(args.pdf)
    print(f"[info] Using doc_id: {computed_id}")

    pages, chunks = process_pdf_to_jsonl(args.pdf, args.out_dir)
    print(f"Wrote:\n  {os.path.abspath(pages)}\n  {os.path.abspath(chunks)}")

if __name__ == "__main__":
    main()
