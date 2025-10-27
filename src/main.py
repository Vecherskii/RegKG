# src/main.py

import typer
from typing_extensions import Annotated
from pathlib import Path

# Import pipeline runner functions
from src.pipeline.emit import process_pdf_to_jsonl # Corrected PDF parser import
from src.pipeline.nlp_extraction import nlp_app # Import NLP app
from src.pipeline.graph_population import graph_app # Import Graph app

# Main Typer application
app = typer.Typer(help="The main control panel for the Regulatory Knowledge Graph pipeline.")

# --- Existing Commands ---

@app.command()
def hello(name: Annotated[str, typer.Argument(help="Your name.")]):
    """
    A simple test command to confirm the environment is working.
    """
    print(f"Hello, {name}! The CLI is working.")
    print("Typer, our control panel, is successfully set up. 🚀")

# Command to run the PDF parsing pipeline
@app.command(name="run-parser")
def run_pdf_parser(
    pdf_path: Path = typer.Option(..., "--pdf", help="Path to the input PDF file."),
    out_dir: Path = typer.Option(..., "--out-dir", help="Directory to save the output JSONL files."),
):
    """
    Processes a single PDF document, extracts structure, and saves pages.jsonl and chunks.jsonl.
    """
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        raise typer.Exit(code=1)
    if not out_dir.exists():
        print(f"Creating output directory: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    elif not out_dir.is_dir():
        print(f"Error: Output path {out_dir} exists but is not a directory.")
        raise typer.Exit(code=1)
    print(f"Starting PDF parsing for: {pdf_path}")
    print(f"Output directory: {out_dir}")
    process_pdf_to_jsonl(pdf_path=str(pdf_path), out_dir=str(out_dir)) # Corrected call
    print("\nPDF parsing complete.")
    print(f"Output files saved in: {out_dir}")

# --- Add Subcommands ---

# Add the NLP commands (e.g., nlp extract-knowledge)
app.add_typer(nlp_app, name="nlp")

# Add the Graph commands (e.g., graph populate)
app.add_typer(graph_app, name="graph")


# --- Entry Point ---

if __name__ == "__main__":
    app()