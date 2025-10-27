
# src/pipeline/nlp_extraction.py
"""
NLP Extraction: From PDF Text Chunks to Raw, Page-Grounded Knowledge
====================================================================

This module implements the **NLP Extraction Stage** of the pipeline. It reads
page-grounded text chunks (produced by the PDF parser into `chunks.jsonl`),
detects domain-relevant entities using a **hybrid NER approach** (spaCy
`EntityRuler` + statistical NER), and extracts *candidate grammatical
relationships* between those entities using dependency parsing.

The output is a JSON Lines file (`extracted_knowledge_v2.jsonl`) containing two
record types per chunk:

- ``entity_classification`` records describing each detected entity span
- ``relationship`` records describing subject–predicate–object triples where
  *both* subject and object are entity spans from the same sentence / chunk

Why this Stage?
---------------
This stage casts a wide net. We avoid early, brittle decisions by gathering
*all* plausible entities and relations with pointers back to the source
(`chunk_id`, character offsets). Downstream stages perform **semantic
filtering** and **FIBO ontology mapping** to convert this raw knowledge into
formal RDF triples for the knowledge graph.

Design Principles
-----------------
- **Hybrid NER:** High-precision domain patterns (via `EntityRuler`) take
  precedence; the statistical model fills remaining gaps.
- **Grounded Outputs:** Every extraction is linked to the originating ``chunk_id``
  and includes exact character offsets for reproducibility.
- **Grammatical Relation Mining:** Use dependency trees to identify candidate
  SVO-style links between recognized entities. Verbs are expanded to their
  multi-word forms (auxiliaries, negations, particles) for better fidelity.

Usage (CLI)
-----------
This file exposes a Typer sub-app, so you can either import and call it from
``src/main.py`` or run directly:

.. code-block:: bash

   # From project root
   python -m src.pipeline.nlp_extraction run \
     --input chunks.jsonl \
     --output extracted_knowledge_v2.jsonl \
     --patterns src/pipeline/config/entity_patterns.jsonl

Schema of Outputs
-----------------
``entity_classification``
    {
      "extraction_id": "<uuid4>",
      "chunk_id": "<chunk id>",
      "type": "entity_classification",
      "text_span": "<exact text>",
      "start_char": <int>,
      "end_char": <int>,
      "entity_label": "<label>",
      "confidence": null
    }

``relationship``
    {
      "extraction_id": "<uuid4>",
      "chunk_id": "<chunk id>",
      "type": "relationship",
      "subject_span": "<text>",
      "subject_label": "<label>",
      "predicate_text": "<verb phrase>",
      "object_span": "<text>",
      "object_label": "<label>",
      "confidence": null
    }

Notes
-----
- Confidence is currently ``null`` because the hybrid approach does not expose a
  unified probability. If you later add a scoring step, you can populate it here.
- This module avoids any ontology-specific mapping on purpose; that is handled
  downstream.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import srsly
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span, Token
import typer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPACY_MODEL: str = "en_core_web_lg"
"""Default spaCy model name. The large English model provides robust vectors,
POS tagging, and dependency parsing which improves RE quality."""

DEFAULT_PATTERNS_PATH: Path = Path("src/pipeline/config/entity_patterns.jsonl")
"""Default location of domain entity patterns consumed by the ``EntityRuler``."""

ALLOWED_CHUNK_TYPES: Set[str] = {
    "paragraph",
    "heading",
    "list_item",
    "caption",
}
"""Chunk block types from the PDF parser to consider for NLP extraction."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityRecord:
    """A detected entity span within a chunk.

    Attributes
    ----------
    extraction_id:
        A unique identifier for this record (``uuid4``).
    chunk_id:
        Identifier of the source chunk from the PDF stage.
    text_span:
        Exact text of the entity as it appears in the chunk.
    start_char, end_char:
        Character offsets within the chunk's plain-text string.
    entity_label:
        Entity type label. May come from the custom ruler (e.g., ``RISK_TYPE``)
        or from the statistical model (e.g., ``ORG``, ``DATE``).
    """
    extraction_id: str
    chunk_id: str
    text_span: str
    start_char: int
    end_char: int
    entity_label: str

    def as_json(self) -> Dict[str, Any]:
        """Serialize to the output JSONL schema with ``type`` and ``confidence``.

        Returns
        -------
        dict
            A JSON-serializable dictionary following the ``entity_classification``
            output schema.
        """
        return {
            "extraction_id": self.extraction_id,
            "chunk_id": self.chunk_id,
            "type": "entity_classification",
            "text_span": self.text_span,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "entity_label": self.entity_label,
            "confidence": None,
        }


@dataclass(frozen=True)
class RelationshipRecord:
    """A candidate grammatical relationship discovered in a chunk sentence.

    Attributes
    ----------
    extraction_id:
        Unique identifier (``uuid4``).
    chunk_id:
        Identifier of the source chunk from the PDF stage.
    subject_span, object_span:
        Exact subject and object surface forms drawn from recognized entities.
    subject_label, object_label:
        Their corresponding entity labels.
    predicate_text:
        The full verb phrase connecting subject and object, including auxiliaries,
        particles, and negation when present.
    """
    extraction_id: str
    chunk_id: str
    subject_span: str
    subject_label: str
    predicate_text: str
    object_span: str
    object_label: str

    def as_json(self) -> Dict[str, Any]:
        """Serialize to the output JSONL schema with ``type`` and ``confidence``.

        Returns
        -------
        dict
            A JSON-serializable dictionary following the ``relationship``
            output schema.
        """
        return {
            "extraction_id": self.extraction_id,
            "chunk_id": self.chunk_id,
            "type": "relationship",
            "subject_span": self.subject_span,
            "subject_label": self.subject_label,
            "predicate_text": self.predicate_text,
            "object_span": self.object_span,
            "object_label": self.object_label,
            "confidence": None,
        }


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class NlpExtractor:
    """Extract entities and grammar-based relationships from text chunks.

    This class encapsulates the spaCy pipeline setup (hybrid NER) and provides
    helpers to process individual chunks or entire JSONL files.

    Parameters
    ----------
    model_name:
        spaCy model to load (defaults to :data:`SPACY_MODEL`).
    patterns_path:
        Path to the JSONL file with ``EntityRuler`` patterns. Each line should
        be a JSON object with the fields accepted by spaCy's EntityRuler
        (e.g., ``{"label": "REGULATOR", "pattern": "BCBS"}``).
    allowed_chunk_types:
        Iterable of block types to process. If ``None``, :data:`ALLOWED_CHUNK_TYPES`
        is used.

    Notes
    -----
    The **EntityRuler** is added *before* the statistical NER and with
    ``overwrite_ents=True`` so that high-precision domain patterns take
    precedence when overlaps occur.
    """

    def __init__(
        self,
        model_name: str = SPACY_MODEL,
        patterns_path: Path = DEFAULT_PATTERNS_PATH,
        allowed_chunk_types: Optional[Iterable[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.patterns_path = Path(patterns_path)
        self.allowed_chunk_types: Set[str] = set(allowed_chunk_types) if allowed_chunk_types else set(ALLOWED_CHUNK_TYPES)
        self.nlp: Language = self._init_nlp()

    # ------------------------------ Pipeline ------------------------------

    def _init_nlp(self) -> Language:
        """Load spaCy, register an ``EntityRuler`` loaded from patterns, and return the pipeline.

        Returns
        -------
        spacy.language.Language
            A configured spaCy pipeline with:
            - Tokenizer
            - Tagger / Parser
            - EntityRuler (before NER) with custom domain labels
            - Statistical NER (``en_core_web_lg``)
        """
        if not spacy.util.is_package(self.model_name) and not Path(self.model_name).exists():
            raise RuntimeError(
                f"spaCy model '{self.model_name}' is not installed or could not be found. "
                f"Install it with: python -m spacy download {self.model_name}"
            )

        nlp = spacy.load(self.model_name)

        # Create an EntityRuler and load patterns
        ruler = nlp.add_pipe("entity_ruler", first=True, config={"overwrite_ents": True})
        patterns = self._load_entity_patterns(self.patterns_path)
        ruler.add_patterns(patterns)

        # Ensure the statistical NER is present (some pipelines may not include it)
        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner", last=True)

        return nlp

    @staticmethod
    def _load_entity_patterns(path: Path) -> List[Dict[str, Any]]:
        """Load EntityRuler patterns from a JSONL file.

        Parameters
        ----------
        path:
            File path to a JSON Lines file. Each line is a JSON object following
            spaCy's expected schema for ruler patterns.

        Returns
        -------
        list of dict
            The list of pattern objects.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is empty or contains no valid patterns.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Entity patterns file not found: {path}")
        patterns = list(srsly.read_jsonl(path))
        if not patterns:
            raise ValueError(f"No patterns found in: {path}")
        return patterns

    # ---------------------------- Chunk helpers ---------------------------

    def _chunk_is_allowed(self, chunk: Dict[str, Any]) -> bool:
        """Return ``True`` if the chunk type is eligible for processing.

        Parameters
        ----------
        chunk:
            A single chunk dictionary from the parser stage. Expected keys include
            ``block_type`` and ``text``.

        Returns
        -------
        bool
            Whether this chunk should be processed.
        """
        return chunk.get("block_type") in self.allowed_chunk_types and bool(chunk.get("text", "").strip())

    def _iter_chunks(self, chunks_path: Path) -> Iterator[Dict[str, Any]]:
        """Yield chunk dictionaries from a JSONL file.

        This is a thin generator wrapper over :func:`srsly.read_jsonl` that
        allows for future pre-filtering or streaming enhancements.

        Parameters
        ----------
        chunks_path:
            Path to the ``chunks.jsonl`` input file.

        Yields
        ------
        dict
            One chunk at a time.
        """
        yield from srsly.read_jsonl(chunks_path)

    # ---------------------------- Entity extraction -----------------------

    def extract_entities(self, doc: Doc, chunk_id: str) -> List[EntityRecord]:
        """Extract entities from a processed :class:`Doc` and tag with ``chunk_id``.

        This method trusts spaCy's merged view of entities after the hybrid
        pipeline runs (domain ruler + statistical NER). Each span is serialized
        with offsets relative to ``doc.text`` (which corresponds to the chunk text).

        Parameters
        ----------
        doc:
            spaCy document produced by :attr:`nlp` from a chunk's text.
        chunk_id:
            Identifier of the originating chunk (used to ground the extraction).

        Returns
        -------
        list of EntityRecord
            All entity spans in the document.
        """
        entities: List[EntityRecord] = []
        for ent in doc.ents:
            entities.append(
                EntityRecord(
                    extraction_id=str(uuid.uuid4()),
                    chunk_id=chunk_id,
                    text_span=ent.text,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    entity_label=ent.label_,
                )
            )
        return entities

    # ------------------------ Relationship extraction ---------------------

    @staticmethod
    def _verb_phrase(head: Token) -> str:
        """Return a readable multi-word verb phrase rooted at ``head``.

        The phrase includes:
        - Negation tokens (``not``, ``n't``)
        - Auxiliary verbs (``has``, ``will``, ``should``)
        - Particles / adpositions that are part of phrasal verbs (``up``, ``out``)

        Parameters
        ----------
        head:
            The verbal head token (typically a token whose POS is ``VERB`` or
            ``AUX``) connecting subject and object in the dependency tree.

        Returns
        -------
        str
            A space-normalized string representing the full verb phrase.

        Notes
        -----
        This function is deliberately permissive; the downstream mapping stage
        can canonicalize verb lemmas if needed.
        """
        tokens: List[Token] = [head]

        # Include auxiliaries and negations from ancestors and children
        def include(tok: Token) -> bool:
            return tok.dep_ in {"aux", "auxpass", "neg", "prt"} or tok.i in getattr(head, "conj_chain", set())

        # Collect relevant children
        children = [t for t in head.children if include(t)]
        tokens.extend(children)

        # Collect relevant ancestors (rare but can help with modal chains)
        ancestors = [t for t in head.ancestors if include(t)]
        tokens.extend(ancestors)

        # Sort by document order and join
        tokens_sorted = sorted(set(tokens), key=lambda t: t.i)
        phrase = " ".join(t.text for t in tokens_sorted)
        return " ".join(phrase.split())  # normalize whitespace

    def extract_relationships(self, doc: Doc, chunk_id: str) -> List[RelationshipRecord]:
        """Extract subject–predicate–object relationships between recognized entities.

        Strategy
        --------
        For each sentence, we traverse tokens to locate verbs that connect a
        nominal subject (``nsubj``/``nsubjpass``) with a direct or prepositional
        object (``dobj``/``pobj``). If both arguments are *entities* as decided
        by the hybrid NER stage, we emit a candidate triple.

        Parameters
        ----------
        doc:
            spaCy document for the current chunk.
        chunk_id:
            Identifier of the originating chunk.

        Returns
        -------
        list of RelationshipRecord
            Candidate relationships discovered in the document.

        Caveats
        -------
        - Cross-sentence relations are **not** considered.
        - We require both ends to be entity spans to keep candidates focused.
        """
        rels: List[RelationshipRecord] = []

        # Build a quick lookup from token index to entity span / label
        ent_by_token: Dict[int, Tuple[str, str]] = {}
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                ent_by_token[i] = (ent.text, ent.label_)

        for sent in doc.sents:
            for token in sent:
                if token.pos_ not in {"VERB", "AUX"}:
                    continue

                # Find subject(s)
                subjects = [child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}]
                if not subjects:
                    continue

                # Candidate objects: direct object or objects of prepositions/governed adpositions
                objects = [child for child in token.children if child.dep_ in {"dobj", "pobj", "attr", "dative"}]

                # Also consider pobj of prepositions attached to the verb
                for prep in (c for c in token.children if c.dep_ == "prep"):
                    objects.extend([pobj for pobj in prep.children if pobj.dep_ == "pobj"])

                if not objects:
                    continue

                # Create relationships for each subj-object pair that map to entities
                for subj in subjects:
                    subj_ent = ent_by_token.get(subj.i)
                    if not subj_ent:
                        continue
                    for obj in objects:
                        obj_ent = ent_by_token.get(obj.i)
                        if not obj_ent:
                            continue

                        predicate = self._verb_phrase(token)
                        rels.append(
                            RelationshipRecord(
                                extraction_id=str(uuid.uuid4()),
                                chunk_id=chunk_id,
                                subject_span=subj_ent[0],
                                subject_label=subj_ent[1],
                                predicate_text=predicate,
                                object_span=obj_ent[0],
                                object_label=obj_ent[1],
                            )
                        )
        return rels

    # --------------------------- High-level API ---------------------------

    def process_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single chunk dictionary into JSONL-ready records.

        Steps
        -----
        1. Skip chunk if its ``block_type`` is not allowed or text is empty.
        2. Run spaCy over the chunk text.
        3. Extract entities and relationships.
        4. Serialize to the output JSON schema.

        Parameters
        ----------
        chunk:
            A chunk dictionary from ``chunks.jsonl``. Required keys:
            - ``chunk_id`` (str): stable identifier used for linkage
            - ``text`` (str): plain text of the chunk
            - ``block_type`` (str): one of the allowed block types

        Returns
        -------
        list of dict
            Zero or more output records (entities + relationships) for this chunk.
        """
        if not self._chunk_is_allowed(chunk):
            return []

        text = chunk.get("text", "")
        if not text.strip():
            return []

        doc = self.nlp.make_doc(text)
        # Ensure sentences and entities are available (parsing & NER)
        with self.nlp.select_pipes(enable=["tok2vec", "tagger", "morphologizer", "parser", "attribute_ruler", "lemmatizer", "ner", "entity_ruler"]):
            doc = self.nlp(doc)

        chunk_id = chunk.get("chunk_id", "")
        if not chunk_id:
            # We keep processing but warn via record to avoid silent data loss.
            chunk_id = "<missing_chunk_id>"

        ents = self.extract_entities(doc, chunk_id)
        rels = self.extract_relationships(doc, chunk_id)

        # Serialize
        records: List[Dict[str, Any]] = [e.as_json() for e in ents] + [r.as_json() for r in rels]
        return records

    def process_file(self, chunks_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process a chunks JSONL file and return all extraction records.

        Parameters
        ----------
        chunks_path:
            Path to the input ``chunks.jsonl``.
        limit:
            Optional limit on the number of chunks to process (useful for testing).

        Returns
        -------
        list of dict
            Combined output records for all processed chunks.
        """
        results: List[Dict[str, Any]] = []
        for i, chunk in enumerate(self._iter_chunks(chunks_path)):
            if limit is not None and i >= limit:
                break
            results.extend(self.process_chunk(chunk))
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_nlp_extraction(
    input: Path = typer.Option(..., "--input", "-i", help="Input chunks.jsonl path"),
    output: Path = typer.Option(..., "--output", "-o", help="Output JSONL path"),
    patterns: Path = typer.Option(DEFAULT_PATTERNS_PATH, "--patterns", "-p", help="EntityRuler patterns JSONL"),
    model: str = typer.Option(SPACY_MODEL, "--model", "-m", help="spaCy model name or local path"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process only the first N chunks (debug)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite output if it exists"),
) -> None:
    """CLI entrypoint: extract entities & relationships and write a JSONL file.

    Parameters
    ----------
    input:
        Path to the ``chunks.jsonl`` file produced by the PDF parsing stage.
    output:
        Path to the destination JSONL file (created if missing).
    patterns:
        Path to the JSONL file with EntityRuler patterns.
    model:
        spaCy model name or path to a local model directory.
    limit:
        Optional limit for the number of chunks to process (for faster iteration).
    overwrite:
        If ``True``, remove the output file if it exists before writing.

    Behavior
    --------
    - Loads the hybrid spaCy pipeline as configured.
    - Streams through input chunks, generating entity and relationship records.
    - Writes all records to the specified JSONL file.
    """
    if output.exists() and not overwrite:
        raise typer.Exit(f"Output already exists: {output}. Use --overwrite to replace.")

    if output.exists() and overwrite:
        output.unlink()

    extractor = NlpExtractor(model_name=model, patterns_path=patterns)
    all_records = extractor.process_file(Path(input), limit=limit)

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)
    srsly.write_jsonl(output, all_records)
    typer.echo(f"Wrote {len(all_records)} records to {output}")

# Typer sub-app for modular integration
nlp_app = typer.Typer(name="nlp", help="NLP extraction commands")
nlp_app.command("extract-knowledge")(run_nlp_extraction)

if __name__ == "__main__":
    # Provide a simple 'run' command when executed directly
    temp = typer.Typer()
    temp.command("run")(run_nlp_extraction)
    temp()
