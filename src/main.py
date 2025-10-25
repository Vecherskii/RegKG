from __future__ import annotations

import argparse
import os
from pipeline.emit import process_pdf_to_jsonl

def main():
    ap = argparse.ArgumentParser(description="PyMuPDF baseline PDF parser -> pages.jsonl / chunks.jsonl")
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--doc-id", default=None, help="Override document ID (defaults to file stem)")
    args = ap.parse_args()

    pages, chunks = process_pdf_to_jsonl(args.pdf, args.out_dir, doc_id=args.doc_id)
    print(f"Wrote:\n  {os.path.abspath(pages)}\n  {os.path.abspath(chunks)}")

if __name__ == "__main__":
    main()
