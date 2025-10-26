# src/pipeline/nlp_extraction.py

import spacy
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span, Token
import typer
import srsly
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Configuration ---
SPACY_MODEL = "en_core_web_lg"
# Default path to the entity patterns file, relative to this script's location
DEFAULT_PATTERNS_PATH = Path(__file__).parent / "config" / "entity_patterns.jsonl"

RELEVANT_CHUNK_TYPES = {"paragraph", "title", "heading", "list_item", "caption"}

# --- Helper Functions ---
def generate_uuid() -> str:
    return str(uuid.uuid4())

def get_verb_phrase(token: Token) -> str:
    verb_phrase_tokens = []
    for ancestor in token.ancestors:
        if ancestor.dep_ in ("aux", "auxpass"):
            verb_phrase_tokens.append(ancestor)
    for child in token.children:
        if child.dep_ == "neg":
            verb_phrase_tokens.append(child)
    verb_phrase_tokens.append(token)
    for child in token.children:
        if child.dep_ == "prt":
             verb_phrase_tokens.append(child)
    verb_phrase_tokens.sort(key=lambda t: t.i)
    return " ".join([t.text for t in verb_phrase_tokens])

# --- Core NLP Extractor Class ---
class NlpExtractor:
    def __init__(self, model_name: str = SPACY_MODEL, patterns_path: Path = DEFAULT_PATTERNS_PATH):
        """
        Initializes the NLP pipeline, loading entity patterns from a file.

        Args:
            model_name: Name of the base spaCy model to load.
            patterns_path: Path to the .jsonl file containing entity patterns.
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"Loaded spaCy model '{model_name}'.")
        except OSError:
            print(f"spaCy model '{model_name}' not found. Downloading...")
            spacy.cli.download(model_name) # type: ignore
            self.nlp = spacy.load(model_name)
            print("Model downloaded and loaded.")

        # Load patterns from the specified file
        if not patterns_path.exists():
             print(f"Warning: Entity patterns file not found at {patterns_path}. EntityRuler will be empty.")
             entity_patterns = []
        else:
            try:
                entity_patterns = list(srsly.read_jsonl(patterns_path))
                print(f"Loaded {len(entity_patterns)} entity patterns from {patterns_path}.")
            except Exception as e:
                print(f"Error loading entity patterns from {patterns_path}: {e}")
                entity_patterns = []


        # Add EntityRuler *before* the default 'ner' component
        config = {"overwrite_ents": True}
        if "entity_ruler" not in self.nlp.pipe_names:
            self.ruler = self.nlp.add_pipe("entity_ruler", before="ner", config=config)
            if entity_patterns:
                self.ruler.add_patterns(entity_patterns)
                print("Added EntityRuler to the pipeline with custom patterns.")
            else:
                 print("EntityRuler added, but no patterns were loaded.")
        else:
            print("EntityRuler already exists in pipeline. Ensure patterns are loaded if needed.")
            # Optionally reload patterns into existing ruler if logic requires it
            # self.ruler = self.nlp.get_pipe("entity_ruler")
            # self.ruler.from_disk(patterns_path) # Or ruler.add_patterns() if appending


    def process_chunk(self, chunk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunk_id = chunk_data.get("chunk_id", "unknown_chunk")
        text = chunk_data.get("text", "")
        chunk_type = chunk_data.get("chunk_type", "")

        if not text or chunk_type not in RELEVANT_CHUNK_TYPES:
            return []

        cleaned_text = text.replace('\n', ' ').strip()
        if not cleaned_text:
            return []

        doc = self.nlp(cleaned_text)
        extractions = []

        # 1. Extract Entities
        for ent in doc.ents:
            extraction = {
                "extraction_id": generate_uuid(),
                "chunk_id": chunk_id,
                "type": "entity_classification",
                "text_span": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "entity_label": ent.label_,
                "confidence": None
            }
            extractions.append(extraction)

        # 2. Extract Raw Relationships (Subject-Verb_Phrase-Object)
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB" and (token.dep_ == "ROOT" or token.dep_ == "xcomp" or (token.dep_ == "auxpass" and token.head.dep_ == "ROOT")):
                    verb_token = token.head if token.dep_ == "auxpass" else token
                    subjects = [child for child in verb_token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    direct_objects = [child for child in verb_token.children if child.dep_ == "dobj"]
                    prep_objects = []
                    for child in verb_token.children:
                        if child.dep_ == "prep":
                            prep_objects.extend([grandchild for grandchild in child.children if grandchild.dep_ == "pobj"])
                    all_objects = direct_objects + prep_objects

                    for subj_token in subjects:
                        subj_ent = self._get_entity_for_token(subj_token, doc.ents)
                        if not subj_ent: continue
                        for obj_token in all_objects:
                            obj_ent = self._get_entity_for_token(obj_token, doc.ents)
                            if not obj_ent: continue
                            if subj_ent == obj_ent: continue
                            predicate_text = get_verb_phrase(token)
                            extraction = {
                                "extraction_id": generate_uuid(),
                                "chunk_id": chunk_id,
                                "type": "relationship",
                                "subject_span": subj_ent.text,
                                "subject_label": subj_ent.label_,
                                "predicate_text": predicate_text,
                                "object_span": obj_ent.text,
                                "object_label": obj_ent.label_,
                                "confidence": None
                            }
                            extractions.append(extraction)
        return extractions

    def _get_entity_for_token(self, token: Token, entities: tuple[Span, ...]) -> Optional[Span]:
        current = token
        while current is not None:
            for ent in entities:
                if ent.start == current.i:
                     if token.i >= ent.start and token.i < ent.end:
                        return ent
            if current == current.head:
                 break
            current = current.head
        for ent in entities:
            if token.i >= ent.start and token.i < ent.end:
                return ent
        return None

# --- Typer command function ---
def run_nlp_extraction(
    input_file: Path = typer.Option(..., "--input-file", "-i", help="Path to the input chunks.jsonl file."),
    output_file: Path = typer.Option(..., "--output-file", "-o", help="Path to the output extracted_knowledge.jsonl file."),
    patterns_file: Path = typer.Option(DEFAULT_PATTERNS_PATH, "--patterns-file", "-p", help="Path to the entity patterns JSONL file."),
    limit: int = typer.Option(-1, "--limit", "-l", help="Limit the number of chunks processed (for testing). -1 means no limit."),
):
    """
    Runs the NLP extraction pipeline on a chunks.jsonl file using patterns from a specified file.
    Reads chunks, extracts entities/relationships, writes to extracted_knowledge.jsonl.
    """
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        raise typer.Exit(code=1)
    if not patterns_file.exists():
         # Changed to warning, allows running with just base model NER
        print(f"Warning: Patterns file not found at {patterns_file}. Proceeding without custom entity patterns.")
        # raise typer.Exit(code=1) # Or exit if patterns are mandatory

    print(f"Starting NLP extraction from: {input_file}")
    print(f"Using entity patterns from: {patterns_file}")
    print(f"Output will be written to: {output_file}")

    # Pass the patterns file path to the extractor
    extractor = NlpExtractor(patterns_path=patterns_file)

    all_extractions: List[Dict[str, Any]] = []
    processed_count = 0
    skipped_count = 0

    print("Processing chunks...")
    for chunk in srsly.read_jsonl(input_file):
        if limit != -1 and processed_count >= limit:
            print(f"Reached processing limit of {limit} chunks.")
            break
        chunk_type = chunk.get("chunk_type", "")
        if chunk_type not in RELEVANT_CHUNK_TYPES:
            skipped_count += 1
            continue
        chunk_extractions = extractor.process_chunk(chunk)
        all_extractions.extend(chunk_extractions)
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count} relevant chunks...")

    print(f"\nFinished processing.")
    print(f"  Processed {processed_count} relevant chunks.")
    print(f"  Skipped {skipped_count} chunks (irrelevant types or no text).")
    print(f"  Found {len(all_extractions)} total extractions (entities + relationships).")

    print(f"Writing extractions to {output_file}...")
    srsly.write_jsonl(output_file, all_extractions)
    print("Done.")

# --- Define a Typer app instance for this module ---
nlp_app = typer.Typer(name="nlp", help="Commands related to Natural Language Processing.")

# --- Register the command function ---
nlp_app.command("extract-knowledge")(run_nlp_extraction)

# --- Entry point for direct execution ---
if __name__ == "__main__":
    temp_app = typer.Typer()
    temp_app.command("run")(run_nlp_extraction)
    temp_app()