# tests/test_graph_population.py

import pytest
from pathlib import Path
import srsly
import tempfile
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, DCTERMS

# Import the functions/classes to test
from src.pipeline.graph_population import populate_graph_from_extractions, INST, MYPROJ_ONT, FIBO_ORG_LDR

# --- Fixtures ---

@pytest.fixture
def sample_extractions_data() -> List[Dict]:
    """Provides sample extraction data similar to extracted_knowledge.jsonl"""
    return [
        {
            "extraction_id": "ent-1", "chunk_id": "doc1#pg=1#t=p#n=1", "type": "entity_classification",
            "text_span": "IRRBB", "start_char": 0, "end_char": 5, "entity_label": "RISK_TYPE", "confidence": None
        },
        {
            "extraction_id": "ent-2", "chunk_id": "doc1#pg=1#t=p#n=1", "type": "entity_classification",
            "text_span": "Basel Committee", "start_char": 20, "end_char": 35, "entity_label": "REGULATOR", "confidence": None
        },
        { # Relationship linking the two entities above (IDs might need adjustment based on real output)
            "extraction_id": "rel-1", "chunk_id": "doc1#pg=1#t=p#n=1", "type": "relationship",
            "subject_span": "Basel Committee", "subject_label": "REGULATOR", "subject_extraction_id": "ent-2", # Assuming IDs link
            "predicate_text": "defines",
            "object_span": "IRRBB", "object_label": "RISK_TYPE", "object_extraction_id": "ent-1", # Assuming IDs link
            "confidence": None
        },
         { # Entity that won't map with current default mappings
            "extraction_id": "ent-3", "chunk_id": "doc1#pg=2#t=h#n=1", "type": "entity_classification",
            "text_span": "Annex 1", "start_char": 0, "end_char": 7, "entity_label": "WORK_OF_ART", "confidence": None
        }
    ]

@pytest.fixture
def temp_extractions_file(sample_extractions_data) -> Path:
    """Creates a temporary JSONL file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
        srsly.write_jsonl(f.name, sample_extractions_data)
        path = Path(f.name)
    yield path
    path.unlink() # Clean up the file after test

# --- Test Cases ---

def test_populate_graph_creates_entities(temp_extractions_file):
    """Test that entity instances are created with correct type, label, source."""
    g = Graph()
    populate_graph_from_extractions(temp_extractions_file, g)

    # Check for IRRBB instance (using SPARQL)
    # Construct expected URI based on generation logic (this might need adjustment)
    expected_irrbb_uri = INST["doc1/RISK_TYPE_IRRBB_doc1#pg=1#t=p#n=1_hash"] # Placeholder hash
    # A more robust test finds the URI via label and type:
    q_irrbb = """
        SELECT ?inst WHERE {
            ?inst rdf:type my-proj:InterestRateRisk ;
                  rdfs:label "IRRBB"@en ;
                  dcterms:source "doc1#pg=1#t=p#n=1" .
        }
    """
    res_irrbb = list(g.query(q_irrbb, initNs={"my-proj": MYPROJ_ONT, "rdfs": RDFS, "dcterms": DCTERMS, "rdf": RDF}))
    assert len(res_irrbb) == 1, "IRRBB instance not found or incorrect"
    irrbb_uri = res_irrbb[0][0] # Get the actual generated URI

    # Check for Basel Committee instance
    q_bcbs = """
        SELECT ?inst WHERE {
            ?inst rdf:type fibo-org-ldr:RegulatoryAgency ;
                  rdfs:label "Basel Committee"@en ;
                  dcterms:source "doc1#pg=1#t=p#n=1" .
        }
    """
    res_bcbs = list(g.query(q_bcbs, initNs={"fibo-org-ldr": FIBO_ORG_LDR, "rdfs": RDFS, "dcterms": DCTERMS, "rdf": RDF}))
    assert len(res_bcbs) == 1, "Basel Committee instance not found or incorrect"
    bcbs_uri = res_bcbs[0][0]

    # Check that unmapped entity was skipped (or handled as expected)
    q_unmapped = """
        SELECT ?inst WHERE {
             ?inst rdfs:label "Annex 1"@en .
        }
    """
    res_unmapped = list(g.query(q_unmapped, initNs={"rdfs": RDFS}))
    assert len(res_unmapped) == 0, "Unmapped entity 'Annex 1' should not have been created by default"


def test_populate_graph_creates_relationships(temp_extractions_file):
    """Test that relationship triples are created between existing instances."""
    g = Graph()
    populate_graph_from_extractions(temp_extractions_file, g)

    # Find the URIs first (could reuse queries from previous test or make more specific)
    irrbb_uri = g.value(predicate=RDFS.label, object=Literal("IRRBB", lang="en"))
    bcbs_uri = g.value(predicate=RDFS.label, object=Literal("Basel Committee", lang="en"))

    assert irrbb_uri is not None, "IRRBB URI not found for relationship test"
    assert bcbs_uri is not None, "BCBS URI not found for relationship test"

    # Check for the 'defines' relationship using the found URIs
    q_rel = """
        ASK WHERE {
            ?subj prop:defines ?obj .
        }
    """
    # Important: Bind the found URIs to variables in the query
    bindings = {'subj': bcbs_uri, 'obj': irrbb_uri}
    res_rel = g.query(q_rel, initNs={"prop": FIBO_FND_REL}, initBindings=bindings) # Use FIBO property directly

    assert bool(res_rel), f"Expected relationship '{bcbs_uri} defines {irrbb_uri}' not found in graph"

# TODO: Add tests for custom ontology loading/saving if needed
# TODO: Add tests for CLI command integration if desired