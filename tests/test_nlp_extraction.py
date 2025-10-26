# tests/test_nlp_extraction.py

import pytest
from pathlib import Path
import srsly
import tempfile

# Import the class/functions you want to test
from src.pipeline.nlp_extraction import NlpExtractor, RELEVANT_CHUNK_TYPES

# --- Fixtures (Optional but good practice) ---

@pytest.fixture(scope="module")
def nlp_extractor() -> NlpExtractor:
    """Fixture to initialize the NlpExtractor once per test module."""
    # Using a smaller model for tests might speed things up if needed,
    # but stick to the main one for accuracy unless tests become too slow.
    # return NlpExtractor(model_name="en_core_web_sm")
    return NlpExtractor() # Uses the default configured model

@pytest.fixture
def sample_chunk_relevant() -> dict:
    """Provides a sample relevant chunk for testing."""
    return {
      "chunk_id": "BCBS_IRRBB_2016_ba40d329#pg=5#t=list_item#n=1",
      "doc_id": "BCBS_IRRBB_2016_ba40d329",
      "page_num": 5,
      "chunk_type": "list_item",
      "text": "1. Interest rate risk in the banking book (IRRBB) is part of the Basel capital framework’s Pillar 2 (Supervisory Review Process) and subject to the Committee’s guidance set out in the 2004 Principles for the management and supervision of interest rate risk (henceforth, the IRR Principles).",
      "bbox": [76.55001831054688, 131.79188537597656, 541.4276733398438, 201.96710205078125],
      "bbox_unit": "pt",
      "page_size": {"w": 595.3200073242188, "h": 842.0399780273438, "unit": "pt"},
      "confidence": 0.9,
      "parser_version": "pymupdf_baseline_v1",
      "text_hash": "sha1-495222de36a1330cadac7356be82e07f836fc404"
    }

@pytest.fixture
def sample_chunk_irrelevant() -> dict:
    """Provides a sample irrelevant chunk for testing."""
    return {
        "chunk_id": "BCBS_IRRBB_2016_ba40d329#pg=1#t=header#n=1",
        "doc_id": "BCBS_IRRBB_2016_ba40d329",
        "page_num": 1,
        "chunk_type": "header", # Not in RELEVANT_CHUNK_TYPES
        "text": "This is a header.",
        "bbox": [0,0,1,1], "bbox_unit": "pt", "page_size": {"w": 1, "h": 1, "unit": "pt"},
        "confidence": 0.75, "parser_version": "pymupdf_baseline_v1", "text_hash": "somehash"
    }

# --- Test Cases ---

def test_nlp_extractor_initialization(nlp_extractor):
    """Test if the NlpExtractor initializes without errors."""
    assert nlp_extractor is not None
    assert nlp_extractor.nlp is not None
    assert "entity_ruler" in nlp_extractor.nlp.pipe_names

def test_process_chunk_irrelevant_type(nlp_extractor, sample_chunk_irrelevant):
    """Test that irrelevant chunk types return no extractions."""
    extractions = nlp_extractor.process_chunk(sample_chunk_irrelevant)
    assert extractions == []

def test_process_chunk_no_text(nlp_extractor):
    """Test that chunks with empty text return no extractions."""
    chunk = {"chunk_id": "test#1", "chunk_type": "paragraph", "text": ""}
    extractions = nlp_extractor.process_chunk(chunk)
    assert extractions == []
    chunk = {"chunk_id": "test#2", "chunk_type": "paragraph", "text": None}
    extractions = nlp_extractor.process_chunk(chunk)
    assert extractions == []

def test_entity_extraction_basic(nlp_extractor, sample_chunk_relevant):
    """Test basic entity recognition using the EntityRuler patterns."""
    extractions = nlp_extractor.process_chunk(sample_chunk_relevant)
    entities = [e for e in extractions if e["type"] == "entity_classification"]

    assert len(entities) > 0 # Check that *some* entities were found

    # Check for specific expected entities based on ENTITY_PATTERNS
    found_irrbb = any(e["text_span"] == "IRRBB" and e["entity_label"] == "RISK_TYPE" for e in entities)
    assert found_irrbb, "Expected entity 'IRRBB' (RISK_TYPE) not found"

    found_risk_phrase = any("Interest rate risk" in e["text_span"] and e["entity_label"] == "RISK_TYPE" for e in entities)
    assert found_risk_phrase, "Expected entity 'Interest rate risk...' (RISK_TYPE) not found"

    found_basel_framework = any("Basel capital framework" in e["text_span"] and e["entity_label"] == "REGULATION_FRAMEWORK" for e in entities)
    assert found_basel_framework, "Expected entity 'Basel capital framework...' (REGULATION_FRAMEWORK) not found"

    found_pillar_2 = any(e["text_span"] == "Pillar 2" and e["entity_label"] == "REGULATION_FRAMEWORK" for e in entities)
    assert found_pillar_2, "Expected entity 'Pillar 2' (REGULATION_FRAMEWORK) not found"

    found_committee = any(e["text_span"] == "Committee" and e["entity_label"] == "REGULATOR" for e in entities)
    assert found_committee, "Expected entity 'Committee' (REGULATOR) not found"

    found_irr_principles = any(e["text_span"] == "IRR Principles" and e["entity_label"] == "REGULATION_NAME" for e in entities)
    assert found_irr_principles, "Expected entity 'IRR Principles' (REGULATION_NAME) not found"


def test_relationship_extraction_placeholder(nlp_extractor, sample_chunk_relevant):
    """
    Placeholder test for relationship extraction.
    This needs to be refined once we inspect the actual output.
    """
    extractions = nlp_extractor.process_chunk(sample_chunk_relevant)
    relationships = [e for e in extractions if e["type"] == "relationship"]

    # For now, just assert that the list exists (even if empty)
    assert isinstance(relationships, list)

    # TODO: Add specific assertions once expected relationships are known.
    # Example (will likely fail initially):
    # expected_rel_found = any(
    #     r["subject_label"] == "REGULATION_NAME" and
    #     r["predicate_text"] == "set out" and # Example predicate
    #     r["object_label"] == "GUIDANCE" # Assuming guidance is found
    #     for r in relationships
    # )
    # assert expected_rel_found, "Did not find expected relationship structure"


# --- Optional: Test the CLI command ---
# These are more like integration tests

# from typer.testing import CliRunner
# from src.main import app # Assuming your main app is in src.main

# runner = CliRunner()

# def test_cli_extract_knowledge_command():
#     """Test the CLI command integration."""
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # Create dummy input file
#         input_path = Path(tmpdir) / "input_chunks.jsonl"
#         sample_data = [{"chunk_id": "c1", "chunk_type": "paragraph", "text": "BCBS issued the IRRBB guidelines."}]
#         srsly.write_jsonl(input_path, sample_data)

#         output_path = Path(tmpdir) / "output_knowledge.jsonl"

#         result = runner.invoke(app, [
#             "nlp", "extract-knowledge",
#             "--input-file", str(input_path),
#             "--output-file", str(output_path),
#         ])

#         print("CLI Output:", result.stdout) # Print output for debugging
#         assert result.exit_code == 0
#         assert output_path.exists()

#         # Check output content (basic)
#         output_data = list(srsly.read_jsonl(output_path))
#         assert len(output_data) > 0 # Check it produced something