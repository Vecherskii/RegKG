# src/pipeline/graph_population.py

import typer
import srsly
import uuid
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Import rdflib components
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, DCTERMS, OWL, XSD, SKOS # Added DCTERMS, SKOS

# --- Configuration ---

# Define Namespaces
MYPROJ_ONT = Namespace("http://my-project.com/ontology/")
INST = Namespace("http://my-project.com/instance/")
FIBO_FND_RISK = Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Risk/Risk/")
FIBO_FND_REL = Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/")
FIBO_ORG_LDR = Namespace("https://spec.edmcouncil.org/fibo/ontology/ORG/Leaders/Leadership/")
FIBO_BE_LE = Namespace("https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/LegalEntity/")
FIBO_BE_LE_CB = Namespace("https://spec.edmcouncil.org/fibo/ontology/BE/LegalEntities/CorporateBodies/")
FIBO_FBC_FI_FI = Namespace("https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstruments/")
FIBO_SEC_SEC_BS = Namespace("https://spec.edmcouncil.org/fibo/ontology/SEC/Securities/Baskets/")

# Initial Mapping Dictionaries
ENTITY_LABEL_TO_CLASS_URI: Dict[str, URIRef] = {
    "RISK_TYPE": MYPROJ_ONT.InterestRateRisk,
    "REGULATOR": FIBO_ORG_LDR.RegulatoryAgency,
    "METRIC": MYPROJ_ONT.RegulatoryMetric,
    "CAPITAL_TIER": FIBO_SEC_SEC_BS.Capital, # Example - Check if FIBO has better class
    "REGULATION_NAME": MYPROJ_ONT.Regulation, # Needs definition
    "FINANCIAL_PRODUCT": FIBO_FBC_FI_FI.FinancialInstrument,
    "PROCESS": MYPROJ_ONT.RegulatoryProcess, # Needs definition
    "REGULATION_FRAMEWORK": MYPROJ_ONT.RegulatoryFramework, # Needs definition
    "COMMITTEE": FIBO_BE_LE_CB.GoverningAuthority,
    "ROLE": MYPROJ_ONT.Role, # Needs definition
    "INSTITUTION_TYPE": FIBO_ORG_LDR.SupervisoryAuthority, # Map "Supervisors" more specifically?
    # TODO: Add specific mapping for 'ORG' label based on text or context if needed
}

# --- EXPANDED PREDICATE MAPPING ---
PREDICATE_TEXT_TO_PROPERTY_URI: Dict[str, URIRef] = {
    # Issuance / Publication
    "issue": FIBO_FND_REL.issues,
    "issued": FIBO_FND_REL.issues,
    "published": FIBO_FND_REL.issues,
    "set out": FIBO_FND_REL.issues, # Mapping to 'issues' for now

    # Requirement / Obligation
    "require": MYPROJ_ONT.requires,
    "requires": MYPROJ_ONT.requires,
    "must": MYPROJ_ONT.requires,
    "must disclose": MYPROJ_ONT.requires, # Map multi-word
    "must ensure": MYPROJ_ONT.requires,
    "must have": MYPROJ_ONT.requires, # Or hasRequirement?
    "must identify": MYPROJ_ONT.requires,
    "must implement": MYPROJ_ONT.requires,
    "must include": MYPROJ_ONT.requires,
    "must publish": MYPROJ_ONT.requires,
    "must be approved": MYPROJ_ONT.requiresApprovalOf, # Needs inverse property/definition
    "must be based on": MYPROJ_ONT.isBasedOn, # Needs definition
    "must be considered": MYPROJ_ONT.requiresConsiderationOf, # Needs definition
    "must be specifically identified": MYPROJ_ONT.requiresIdentificationOf, # Needs definition
    "must be subject to": MYPROJ_ONT.isSubjectTo, # Needs definition

    # Definition
    "define": FIBO_FND_REL.defines,
    "defines": FIBO_FND_REL.defines,
    "describe": SKOS.definition,
    "describes": SKOS.definition,
    "refer": RDFS.seeAlso,
    "refers": RDFS.seeAlso,
    "refers to": RDFS.seeAlso,

    # Composition / Inclusion
    "include": DCTERMS.hasPart,
    "includes": DCTERMS.hasPart,
    "comprise": DCTERMS.hasPart,
    "contain": DCTERMS.hasPart,
    "contains": DCTERMS.hasPart,
    "is part of": DCTERMS.isPartOf,
    "fall into": DCTERMS.isPartOf, # Tentative mapping

    # Application / Scope
    "apply": MYPROJ_ONT.appliesTo, # Needs definition
    "applies to": MYPROJ_ONT.appliesTo,
    "cover": MYPROJ_ONT.appliesTo,
    "covers": MYPROJ_ONT.appliesTo,
    "be subject to": MYPROJ_ONT.isSubjectTo, # Needs definition
    "is subject to": MYPROJ_ONT.isSubjectTo,

    # Causation / Derivation
    "arise from": MYPROJ_ONT.arisesFrom, # Needs definition
    "arises from": MYPROJ_ONT.arisesFrom,
    "be driven by": MYPROJ_ONT.arisesFrom,
    "result in": MYPROJ_ONT.resultsIn, # Needs definition

    # Governance / Oversight
    "oversee": FIBO_FND_REL.manages, # Using manages for now
    "oversees": FIBO_FND_REL.manages,
    "manage": FIBO_FND_REL.manages,
    "manages": FIBO_FND_REL.manages,
    "managing": FIBO_FND_REL.manages,
    "monitor": MYPROJ_ONT.monitors, # Needs definition
    "monitoring": MYPROJ_ONT.monitors,
    "control": MYPROJ_ONT.controls, # Needs definition
    "controls": MYPROJ_ONT.controls,
    "approve": MYPROJ_ONT.approves, # Needs definition
    "approved": MYPROJ_ONT.approves, # Mapping past tense - might need review

    # Other Potentially Useful
    "allow": MYPROJ_ONT.allows, # Needs definition
    "capture": MYPROJ_ONT.captures, # Needs definition
    "captures": MYPROJ_ONT.captures,
    "reflect": MYPROJ_ONT.reflects, # Needs definition
    "reflects": MYPROJ_ONT.reflects,
    "represent": MYPROJ_ONT.represents, # Needs definition
    "represents": MYPROJ_ONT.represents,
    "follow": MYPROJ_ONT.follows, # Needs definition
    "use": MYPROJ_ONT.uses, # Needs definition
    "uses": MYPROJ_ONT.uses,
    "need": MYPROJ_ONT.requires, # Mapping 'need' to requires
    "needs": MYPROJ_ONT.requires,

    # Added from list (potential mappings - review needed)
    "account for": MYPROJ_ONT.accountsFor, # Needs definition
    "affect": MYPROJ_ONT.affects, # Needs definition
    "enable": MYPROJ_ONT.enables, # Needs definition
    "ensure": MYPROJ_ONT.ensures, # Needs definition
    "establish": MYPROJ_ONT.establishes, # Needs definition
    "limit": MYPROJ_ONT.limits, # Needs definition
    "promote": MYPROJ_ONT.promotes, # Needs definition
    "provide": MYPROJ_ONT.provides, # Needs definition
    "provides": MYPROJ_ONT.provides,
}

# --- Helper Functions ---
# ... (generate_instance_uri, _clean_for_uri remain the same) ...
def _clean_for_uri(text: str) -> str:
    text = re.sub(r'[<>\"{}|^`\\]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text[:50]

def generate_instance_uri(doc_id: str, chunk_id: str, text_span: str, entity_label: str) -> URIRef:
    clean_span = _clean_for_uri(text_span)
    chunk_hash = hashlib.sha1(chunk_id.encode()).hexdigest()[:8]
    instance_name = f"{doc_id}/{entity_label}_{clean_span}_{chunk_hash}"
    # Ensure invalid characters for URI path segments are handled (e.g., '/')
    instance_name = instance_name.replace('/', '_') # Basic replacement
    return INST[instance_name]

# --- Core Graph Population Logic ---
def populate_graph_from_extractions(
    extractions_file: Path,
    instance_graph: Graph,
    custom_ontology_graph: Optional[Graph] = None
):
    instance_cache: Dict[str, URIRef] = {}
    processed_extraction_ids: Set[str] = set()

    print(f"Processing extractions from: {extractions_file}")
    extraction_data = list(srsly.read_jsonl(extractions_file))

    # --- Pass 1: Create Entities ---
    print("Pass 1: Creating entity instances...")
    entity_count = 0
    skipped_entity_labels = set() # Track unmapped labels
    for item in extraction_data:
        if item.get("type") == "entity_classification":
            extraction_id = item.get("extraction_id")
            if not extraction_id: continue

            entity_label = item.get("entity_label")
            text_span = item.get("text_span")
            chunk_id = item.get("chunk_id")
            doc_id = chunk_id.split('#')[0] if chunk_id else "unknown_doc"

            if not entity_label or not text_span or not chunk_id: continue

            class_uri = ENTITY_LABEL_TO_CLASS_URI.get(entity_label)
            if class_uri:
                # Only process if not already processed (handles duplicate entity extractions if any)
                if extraction_id not in processed_extraction_ids:
                    instance_uri = generate_instance_uri(doc_id, chunk_id, text_span, entity_label)
                    instance_cache[extraction_id] = instance_uri
                    processed_extraction_ids.add(extraction_id)

                    instance_graph.add((instance_uri, RDF.type, class_uri))
                    instance_graph.add((instance_uri, RDFS.label, Literal(text_span, lang="en")))
                    instance_graph.add((instance_uri, DCTERMS.source, Literal(chunk_id)))
                    entity_count += 1
            else:
                 skipped_entity_labels.add(entity_label) # Track unmapped labels


    print(f"Created {entity_count} entity instances.")
    if skipped_entity_labels:
        print(f"Skipped entities with unmapped labels: {', '.join(sorted(list(skipped_entity_labels)))}")


    # --- Pass 2: Create Relationships ---
    print("Pass 2: Creating relationship triples...")
    rel_count = 0
    skipped_rels_uri = 0
    skipped_rels_prop = 0
    unmapped_predicates = set() # Track unmapped predicates

    for item in extraction_data:
         if item.get("type") == "relationship":
            rel_extraction_id = item.get("extraction_id")
            subj_extraction_id = item.get("subject_extraction_id")
            obj_extraction_id = item.get("object_extraction_id")
            predicate_text = item.get("predicate_text", "").lower().strip() # Normalize

            if not rel_extraction_id or not subj_extraction_id or not obj_extraction_id:
                skipped_rels_uri += 1 # Count if IDs are missing in input
                continue

            # Look up URIs from cache
            subj_uri = instance_cache.get(subj_extraction_id)
            obj_uri = instance_cache.get(obj_extraction_id)

            # Look up property URI
            prop_uri = PREDICATE_TEXT_TO_PROPERTY_URI.get(predicate_text)

            if subj_uri and obj_uri:
                if prop_uri:
                    instance_graph.add((subj_uri, prop_uri, obj_uri))
                    rel_count += 1
                else:
                    skipped_rels_prop += 1 # Count if property mapping failed
                    unmapped_predicates.add(item.get("predicate_text", "")) # Original case for reporting
            else:
                skipped_rels_uri += 1 # Count if subject/object URI lookup failed

    print(f"Created {rel_count} relationship triples.")
    if skipped_rels_uri > 0:
        print(f"Skipped {skipped_rels_uri} relationships due to missing subject/object URI in cache or input.")
    if skipped_rels_prop > 0:
        print(f"Skipped {skipped_rels_prop} relationships due to unmapped predicate text.")
        if unmapped_predicates:
             print(f"  Unmapped predicates encountered: {', '.join(sorted(list(unmapped_predicates)))}")


# --- Typer command function ---
# ... (run_graph_population remains largely the same, just ensure bindings are added) ...
def run_graph_population(
    input_file: Path = typer.Option(..., "--input-file", "-i", help="Path to the input extracted_knowledge.jsonl file."),
    output_ontology_file: Path = typer.Option(Path("ontology/my_project_ontology.ttl"), "--output-ontology", "-co", help="Path to save the custom ontology definitions."),
    output_triples_file: Path = typer.Option(..., "--output-triples", "-o", help="Path to save the generated knowledge graph instance triples (.ttl)."),
):
    """
    Processes NLP extractions, maps them to an ontology, and generates RDF triples.
    """
    if not input_file.exists():
        print(f"Error: Input file not found at '{input_file}'")
        raise typer.Exit(code=1)

    output_ontology_file.parent.mkdir(parents=True, exist_ok=True)
    output_triples_file.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing graphs...")
    custom_onto_graph = Graph()
    # Bind prefixes for custom ontology file
    custom_onto_graph.bind("my-proj", MYPROJ_ONT)
    custom_onto_graph.bind("fibo-fnd-risk-risk", FIBO_FND_RISK)
    custom_onto_graph.bind("fibo-fnd-rel-rel", FIBO_FND_REL)
    custom_onto_graph.bind("owl", OWL)
    custom_onto_graph.bind("rdfs", RDFS)
    custom_onto_graph.bind("skos", SKOS)
    custom_onto_graph.bind("dcterms", DCTERMS) # Added binding

    # --- Add initial custom definitions ---
    # (Copied from previous step, but should match definitions in my_project_ontology.ttl)
    custom_onto_graph.add((MYPROJ_ONT.InterestRateRisk, RDF.type, OWL.Class))
    custom_onto_graph.add((MYPROJ_ONT.InterestRateRisk, RDFS.subClassOf, FIBO_FND_RISK.Risk))
    custom_onto_graph.add((MYPROJ_ONT.InterestRateRisk, RDFS.label, Literal("Interest Rate Risk", lang="en")))
    # ... Add ALL definitions for classes/properties used in mappings here ...
    # Example for a newly mapped property's class (if needed):
    custom_onto_graph.add((MYPROJ_ONT.Regulation, RDF.type, OWL.Class)) # If REGULATION_NAME maps to this
    custom_onto_graph.add((MYPROJ_ONT.Regulation, RDFS.label, Literal("Regulation", lang="en")))
    # Add definitions for ALL custom properties used in PREDICATE_TEXT_TO_PROPERTY_URI
    # (Example: my-proj:appliesTo, my-proj:arisesFrom, etc.)


    instance_graph = Graph()
    # Bind prefixes for instance data file readability
    instance_graph.bind("my-proj", MYPROJ_ONT)
    instance_graph.bind("inst", INST)
    instance_graph.bind("dcterms", DCTERMS) # Added
    instance_graph.bind("rdf", RDF)
    instance_graph.bind("rdfs", RDFS)
    instance_graph.bind("skos", SKOS) # Added
    # -- Add bindings for ALL used FIBO namespaces --
    instance_graph.bind("fibo-fnd-risk-risk", FIBO_FND_RISK)
    instance_graph.bind("fibo-fnd-rel-rel", FIBO_FND_REL)
    instance_graph.bind("fibo-org-ldr", FIBO_ORG_LDR)
    instance_graph.bind("fibo-be-le", FIBO_BE_LE)
    instance_graph.bind("fibo-be-le-cb", FIBO_BE_LE_CB)
    instance_graph.bind("fibo-fbc-fi-fi", FIBO_FBC_FI_FI)
    instance_graph.bind("fibo-sec-sec-bs", FIBO_SEC_SEC_BS)
    # ---

    populate_graph_from_extractions(input_file, instance_graph, custom_onto_graph)

    print(f"Serializing custom ontology to '{output_ontology_file}'...")
    try:
        # Save the definitions added within this script (useful for consistency)
        # For production, load/update the actual ontology file instead of redefining here
        custom_onto_graph.serialize(destination=str(output_ontology_file), format="turtle")
    except Exception as e:
        print(f"Error serializing custom ontology: {e}")

    print(f"Serializing instance triples to '{output_triples_file}'...")
    try:
        instance_graph.serialize(destination=str(output_triples_file), format="turtle")
    except Exception as e:
        print(f"Error serializing instance triples: {e}")

    print("Graph population complete.")

# --- Define Typer app and register command ---
graph_app = typer.Typer(name="graph", help="Commands related to Knowledge Graph population.")
graph_app.command("populate")(run_graph_population)

# --- Entry point for direct execution ---
if __name__ == "__main__":
    temp_app = typer.Typer()
    temp_app.command("run")(run_graph_population)
    temp_app()