
# src/pipeline/graph_population_semantic.py
# (short header) — See module docstring below for full explanation.
"""
Knowledge Formalization & Graph Population (Semantic Normalization Layer)
<snip: full detailed docstring retained in previous message>
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import srsly
import typer
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, DCTERMS, OWL, XSD, SKOS

NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")

def _as_decimal_literal(text: str) -> Optional[Literal]:
    t = (text or "").strip()
    # Strip trailing percent signs and remember to scale if you like; for now we just drop the sign.
    if t.endswith("%"):
        t = t[:-1].strip()
    if NUMERIC_RE.match(t):
        return Literal(t, datatype=XSD.decimal)
    return None

FIBO_FND = Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/")
FIBO_FND_REL = Namespace("https://spec.edmcouncil.org/fibo/ontology/FND/Relations/Relations/")
FIBO_BE = Namespace("https://spec.edmcouncil.org/fibo/ontology/BE/")
MY = Namespace("https://w3id.org/my-project/ontology/")
INST = Namespace("https://w3id.org/my-project/id/")

ENTITY_LABEL_TO_CLASS_URI: Dict[str, str] = {
    "RISK_TYPE": str(FIBO_FND["Risk/FinancialRisk"]),
    "REGULATOR": str(FIBO_BE["GovernmentEntities/RegulatoryAgency"]),
    "REGULATION_NAME": str(FIBO_FND["Agreements/Contracts/Contract"]),
    "CAPITAL_TIER": str(FIBO_FND["Accounting/CurrencyAmount"]),
    "FINANCIAL_PRODUCT": str(FIBO_FND["ProductsAndServices/FinancialProduct"]),
    "FINANCIAL_INSTRUMENT": str(FIBO_FND["Accounting/FinancialInstruments/FinancialInstrument"]),
}

PREDICATE_CANONICAL_TO_PROPERTY_URI: Dict[str, str] = {
    "define": str(FIBO_FND_REL["defines"]),
    "publish": str(DCTERMS.issued),
    "issue": str(FIBO_FND_REL["issues"]),
    "require": str(FIBO_FND_REL["requires"]),
    "apply": str(FIBO_FND_REL["appliesTo"]),
    "govern": str(FIBO_FND_REL["governs"]),
    "refer": str(FIBO_FND_REL["references"]),
    "apply::p(to)": str(FIBO_FND_REL["appliesTo"]),
    "refer::p(to)": str(FIBO_FND_REL["references"]),
    "arise::p(from)": str(FIBO_FND_REL["isEvidencedBy"]),
    "expose::p(to)": str(FIBO_FND_REL["isExposedTo"]),
}

AUXILIARIES: Set[str] = {
    "be","am","is","are","was","were","been","being",
    "do","does","did","done","doing",
    "have","has","had","having",
    "will","would","shall","should","can","could","may","might","must"
}
NEG_TOKENS: Set[str] = {"not", "n't", "no"}
PREP_TOKENS: Tuple[str, ...] = ("of","to","for","from","in","on","by","with","without","under","over","within","into","onto")
_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class EntityExtraction:
    extraction_id: str
    chunk_id: str
    text_span: str
    entity_label: str

@dataclass(frozen=True)
class RelationshipExtraction:
    extraction_id: str
    chunk_id: str
    subject_extraction_id: Optional[str]
    object_extraction_id: Optional[str]
    subject_span: str
    subject_label: str
    predicate_text: str
    object_span: str
    object_label: str

def _hash_id(*parts: str, prefix: str = "") -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8")); h.update(b"|")
    return prefix + h.hexdigest()[:16]

def mint_instance_uri(extraction: EntityExtraction) -> URIRef:
    slug = _hash_id(extraction.entity_label, extraction.text_span, extraction.chunk_id, extraction.extraction_id, prefix="ent_")
    return INST[slug]

def _simple_verb_lemma(tok: str) -> str:
    """
    Tiny, rule-based lemmatizer for verbs. Carefully avoids stripping 's'
    when the word ends with 'ss' (fixes 'assess' → 'assess', not 'asses').
    """
    t = tok.lower()
    if t in AUXILIARIES:
        return ""

    # Order matters
    rules = (
        ("izing", "ize"),
        ("ising", "ise"),
        ("ing", ""),       # making -> mak (rough but ok for canonical keys)
        ("ies", "y"),      # studies -> study
        ("sses", "ss"),    # passes -> pass (handled before the generic 's')
        ("ed", ""),        # required -> requir (rough but ok)
    )
    for suf, repl in rules:
        if t.endswith(suf) and len(t) > len(suf) + 1:
            t = t[: -len(suf)] + repl
            break

    # The generic 's' rule must NOT fire on words ending with 'ss'
    if t.endswith("s") and not t.endswith("ss") and len(t) > 2:
        t = t[:-1]

    # common artifact fix
    if t.endswith("i") and len(t) > 2:
        t = t[:-1] + "y"

    return t

def canonicalize_predicate(raw: str) -> Tuple[str, str]:
    """
    Compute a canonical predicate key and head lemma from a raw verb phrase.

    Returns (canonical_key, head_lemma).
    If we cannot extract any alphabetic token, returns ("", "") as a noise guard.
    """
    s = (raw or "").strip().lower()
    toks = _WORD_RE.findall(s)

    # noise guard: non-alphabetic garbage (e.g., weird unicode math letters not in A-Za-z)
    if not toks:
        return "", ""

    neg = any(t in NEG_TOKENS for t in toks)
    core = [t for t in toks if t not in AUXILIARIES and t not in NEG_TOKENS]

    head = ""
    if core:
        head = _simple_verb_lemma(core[-1]) or _simple_verb_lemma(core[0])
    if not head and core:
        head = core[-1]

    # If we still don't have a usable head, treat as noise
    if not head:
        return "", ""

    # detect simple preposition following head in raw text
    prep_found = ""
    for p in PREP_TOKENS:
        if re.search(rf"\b{re.escape(head)}\b\s+{p}\b", s):
            prep_found = p
            break

    parts = [head]
    if prep_found:
        parts.append(f"p({prep_found})")
    if neg:
        parts.append("neg")

    canonical = "::".join(parts)
    return canonical, head

@dataclass
class Telemetry:
    entity_counts: Dict[str, int]
    entity_mapped_counts: Dict[str, int]
    predicate_raw_counts: Dict[str, int]
    predicate_canonical_counts: Dict[str, int]
    predicate_mapped_counts: Dict[str, int]
    relation_outcomes: List[Dict[str, Any]]
    skipped_relations: int
    def to_json(self) -> Dict[str, Any]:
        return {
            "entity_counts": self.entity_counts,
            "entity_mapped_counts": self.entity_mapped_counts,
            "predicate_raw_counts": self.predicate_raw_counts,
            "predicate_canonical_counts": self.predicate_canonical_counts,
            "predicate_mapped_counts": self.predicate_mapped_counts,
            "skipped_relations": self.skipped_relations,
            "relation_outcomes_sample": self.relation_outcomes[:1000],
        }

def load_overrides(path: Optional[Path]) -> Dict[str, str]:
    if not path: return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Mapping file must be a JSON object: {path}")
    return {str(k): str(v) for k,v in data.items()}

def build_entity_class_map(override: Optional[Path]) -> Dict[str, URIRef]:
    base = {k: URIRef(v) for k,v in ENTITY_LABEL_TO_CLASS_URI.items()}
    base.update({k: URIRef(v) for k,v in load_overrides(override).items()})
    return base

def build_predicate_map(override: Optional[Path]) -> Dict[str, URIRef]:
    base = {k: URIRef(v) for k,v in PREDICATE_CANONICAL_TO_PROPERTY_URI.items()}
    base.update({k: URIRef(v) for k,v in load_overrides(override).items()})
    return base

@dataclass
class PopulationConfig:
    input_path: Path
    output_triples: Path
    entity_map_path: Optional[Path]
    predicate_map_path: Optional[Path]
    telemetry_json: Optional[Path]
    telemetry_csv: Optional[Path]
    base_ns: Namespace
    my_ns: Namespace
    inst_ns: Namespace
    repair_by_text: bool = False

def run_population(cfg: PopulationConfig) -> None:
    """
    End-to-end population with semantic normalization, graceful fallbacks, and telemetry.
    - Pass 1: materialize entity instances (mapped class or my:PendingEntity), attach provenance.
    - Pass 2: normalize predicates, map to properties (with canonical fallback), repair/mint entities
      when IDs are missing, emit object-property triples or (for numeric behavior) a datatype triple,
      and reify raw predicate text when property is my:relatedToRaw.
    - Emit TTL + JSON/CSV telemetry.
    """
    # --- Build mappings & graph ---
    ent_map = build_entity_class_map(cfg.entity_map_path)
    pred_map = build_predicate_map(cfg.predicate_map_path)

    g = Graph()
    g.bind("rdf", RDF); g.bind("rdfs", RDFS); g.bind("dcterms", DCTERMS)
    g.bind("owl", OWL); g.bind("xsd", XSD); g.bind("skos", SKOS)
    g.bind("fibo-fnd", FIBO_FND); g.bind("fibo-rel", FIBO_FND_REL); g.bind("fibo-be", FIBO_BE)
    g.bind("my", cfg.my_ns); g.bind("inst", cfg.inst_ns)

    tel = Telemetry(
        entity_counts={},
        entity_mapped_counts={},
        predicate_raw_counts={},
        predicate_canonical_counts={},
        predicate_mapped_counts={},
        relation_outcomes=[],
        skipped_relations=0,
    )

    # --- Load extractions from JSONL ---
    entities: Dict[str, EntityExtraction] = {}
    relationships: List[RelationshipExtraction] = []
    for rec in srsly.read_jsonl(cfg.input_path):
        rtype = rec.get("type")
        if rtype == "entity_classification":
            ee = EntityExtraction(
                extraction_id=rec["extraction_id"],
                chunk_id=rec["chunk_id"],
                text_span=rec["text_span"],
                entity_label=rec["entity_label"],
            )
            entities[ee.extraction_id] = ee
            tel.entity_counts[ee.entity_label] = tel.entity_counts.get(ee.entity_label, 0) + 1
        elif rtype == "relationship":
            relationships.append(
                RelationshipExtraction(
                    extraction_id=rec["extraction_id"],
                    chunk_id=rec["chunk_id"],
                    subject_extraction_id=rec.get("subject_extraction_id"),
                    object_extraction_id=rec.get("object_extraction_id"),
                    subject_span=rec["subject_span"],
                    subject_label=rec["subject_label"],
                    predicate_text=rec["predicate_text"],
                    object_span=rec["object_span"],
                    object_label=rec["object_label"],
                )
            )

    # --- Pass 1: materialize entities & build lookup indexes ---
    id_to_uri: Dict[str, URIRef] = {}
    text_index: Dict[Tuple[str, str], str] = {}  # (chunk_id, text_span) -> extraction_id

    for ee in entities.values():
        class_uri = ent_map.get(ee.entity_label)
        mapped = class_uri is not None
        if not mapped:
            class_uri = cfg.my_ns["PendingEntity"]

        inst_uri = mint_instance_uri(ee)
        id_to_uri[ee.extraction_id] = inst_uri
        text_index[(ee.chunk_id, ee.text_span)] = ee.extraction_id

        g.add((inst_uri, RDF.type, URIRef(class_uri)))
        g.add((inst_uri, RDFS.label, Literal(ee.text_span)))
        g.add((inst_uri, DCTERMS.source, Literal(ee.chunk_id)))
        g.add((inst_uri, cfg.my_ns["extractionId"], Literal(ee.extraction_id)))
        if not mapped:
            g.add((inst_uri, cfg.my_ns["unmappedLabel"], Literal(ee.entity_label)))
        else:
            tel.entity_mapped_counts[ee.entity_label] = tel.entity_mapped_counts.get(ee.entity_label, 0) + 1

    # --- Pass 2: relations with canonicalization, mapping, repair & fallbacks ---
    for rr in relationships:
        raw = rr.predicate_text or ""
        tel.predicate_raw_counts[raw] = tel.predicate_raw_counts.get(raw, 0) + 1

        canonical, head = canonicalize_predicate(raw)
        tel.predicate_canonical_counts[canonical] = tel.predicate_canonical_counts.get(canonical, 0) + 1

        prop_uri = pred_map.get(canonical) or pred_map.get(head)
        if prop_uri is None or str(prop_uri).strip() == "":
            prop_uri = cfg.my_ns["relatedToRaw"]
        else:
            tel.predicate_mapped_counts[str(prop_uri)] = tel.predicate_mapped_counts.get(str(prop_uri), 0) + 1

        # Resolve subject/object instance URIs
        sub_uri = id_to_uri.get(rr.subject_extraction_id) if rr.subject_extraction_id else None
        obj_uri = id_to_uri.get(rr.object_extraction_id) if rr.object_extraction_id else None
        repairs: List[str] = []

        # Optional repair by (chunk_id, text_span)
        if cfg.repair_by_text:
            if sub_uri is None and rr.subject_span:
                sid = text_index.get((rr.chunk_id, rr.subject_span))
                if sid and sid in id_to_uri:
                    sub_uri = id_to_uri[sid]
                    repairs.append("repaired_subject_by_text")
            if obj_uri is None and rr.object_span:
                oid = text_index.get((rr.chunk_id, rr.object_span))
                if oid and oid in id_to_uri:
                    obj_uri = id_to_uri[oid]
                    repairs.append("repaired_object_by_text")

        # Mint pending entities if still missing
        if sub_uri is None:
            s_ee = EntityExtraction(str(uuid.uuid4()), rr.chunk_id, rr.subject_span, rr.subject_label)
            sub_uri = mint_instance_uri(s_ee)
            g.add((sub_uri, RDF.type, cfg.my_ns["PendingEntity"]))
            g.add((sub_uri, RDFS.label, Literal(rr.subject_span)))
            g.add((sub_uri, DCTERMS.source, Literal(rr.chunk_id)))
            g.add((sub_uri, cfg.my_ns["unmappedLabel"], Literal(rr.subject_label)))
            repairs.append("minted_pending_subject")

        if obj_uri is None:
            o_ee = EntityExtraction(str(uuid.uuid4()), rr.chunk_id, rr.object_span, rr.object_label)
            obj_uri = mint_instance_uri(o_ee)
            g.add((obj_uri, RDF.type, cfg.my_ns["PendingEntity"]))
            g.add((obj_uri, RDFS.label, Literal(rr.object_span)))
            g.add((obj_uri, DCTERMS.source, Literal(rr.chunk_id)))
            g.add((obj_uri, cfg.my_ns["unmappedLabel"], Literal(rr.object_label)))
            repairs.append("minted_pending_object")

        # --- Special-case: numeric behavior -> datatype property instead of object property ---
        is_behave = (head == "behave") or (str(prop_uri).endswith("/exhibitsBehavior"))
        numeric_lit = _as_decimal_literal(rr.object_span)
        is_numeric_label = rr.object_label in {"PERCENT", "CARDINAL"}

        if is_behave and (numeric_lit is not None or is_numeric_label):
            # Emit typed literal on subject
            data_prop = cfg.my_ns["hasBehaviorParameter"]
            g.add((sub_uri, data_prop, numeric_lit or Literal(rr.object_span)))

            # Reify provenance of the transformation
            st = BNode()
            g.add((st, RDF.type, cfg.my_ns["RawRelation"]))
            g.add((st, RDF.subject, sub_uri))
            g.add((st, RDF.predicate, data_prop))
            g.add((st, RDF.object, (numeric_lit or Literal(rr.object_span))))
            g.add((st, cfg.my_ns["rawPredicateText"], Literal(raw)))
            g.add((st, cfg.my_ns["canonicalPredicate"], Literal(canonical)))

            tel.relation_outcomes.append({
                "relationship_extraction_id": rr.extraction_id,
                "chunk_id": rr.chunk_id,
                "subject_uri": str(sub_uri),
                "object_uri": "",  # literal
                "subject_label": rr.subject_label,
                "object_label": rr.object_label,
                "predicate_raw": raw,
                "predicate_canonical": canonical,
                "property_uri": str(data_prop),
                "repairs": ";".join(repairs) if repairs else "numeric_behavior_literal",
            })
            continue
        # --- end special-case ---

        # Standard object-property triple
        g.add((sub_uri, URIRef(prop_uri), obj_uri))

        # Reify if using raw fallback property
        if str(prop_uri) == str(cfg.my_ns["relatedToRaw"]):
            st = BNode()
            g.add((st, RDF.type, cfg.my_ns["RawRelation"]))
            g.add((st, RDF.subject, sub_uri))
            g.add((st, RDF.predicate, URIRef(prop_uri)))
            g.add((st, RDF.object, obj_uri))
            g.add((st, cfg.my_ns["rawPredicateText"], Literal(raw)))
            g.add((st, cfg.my_ns["canonicalPredicate"], Literal(canonical)))

        # Telemetry row
        tel.relation_outcomes.append({
            "relationship_extraction_id": rr.extraction_id,
            "chunk_id": rr.chunk_id,
            "subject_uri": str(sub_uri),
            "object_uri": str(obj_uri),
            "subject_label": rr.subject_label,
            "object_label": rr.object_label,
            "predicate_raw": raw,
            "predicate_canonical": canonical,
            "property_uri": str(prop_uri),
            "repairs": ";".join(repairs) if repairs else "",
        })

    # --- Serialize outputs ---
    cfg.output_triples.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(cfg.output_triples), format="turtle")

    if cfg.telemetry_json:
        cfg.telemetry_json.parent.mkdir(parents=True, exist_ok=True)
        srsly.write_json(cfg.telemetry_json, {
            "entity_counts": tel.entity_counts,
            "entity_mapped_counts": tel.entity_mapped_counts,
            "predicate_raw_counts": tel.predicate_raw_counts,
            "predicate_canonical_counts": tel.predicate_canonical_counts,
            "predicate_mapped_counts": tel.predicate_mapped_counts,
            "skipped_relations": tel.skipped_relations,
        })

    if cfg.telemetry_csv:
        cfg.telemetry_csv.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "relationship_extraction_id","chunk_id","subject_uri","object_uri",
            "subject_label","object_label","predicate_raw","predicate_canonical",
            "property_uri","repairs"
        ]
        with cfg.telemetry_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for row in tel.relation_outcomes:
                w.writerow({h: row.get(h, "") for h in headers})


app = typer.Typer(name="graph-semantic", help="Knowledge Graph population with semantic normalization & telemetry.")

@app.command("run")
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Path to extracted_knowledge_v*.jsonl"),
    output_triples: Path = typer.Option(..., "--output-triples", "-o", help="Path to output Turtle file"),
    entity_map: Optional[Path] = typer.Option(None, "--entity-map", help="Optional JSON mapping: entity_label -> class URI"),
    predicate_map: Optional[Path] = typer.Option(None, "--predicate-map", help="Optional JSON mapping: canonical predicate -> property URI"),
    telemetry_json: Optional[Path] = typer.Option(None, "--telemetry-json", help="Optional JSON telemetry path"),
    telemetry_csv: Optional[Path] = typer.Option(None, "--telemetry-csv", help="Optional CSV telemetry path"),
    repair_by_text: bool = typer.Option(False, "--repair-by-text", help="Try to repair missing subject/object by (chunk_id, text_span)"),
):
    cfg = PopulationConfig(
        input_path=input,
        output_triples=output_triples,
        entity_map_path=entity_map,
        predicate_map_path=predicate_map,
        telemetry_json=telemetry_json,
        telemetry_csv=telemetry_csv,
        base_ns=FIBO_FND,
        my_ns=MY,
        inst_ns=INST,
        repair_by_text=repair_by_text,
    )
    run_population(cfg)
    typer.echo(f"✅ Wrote triples -> {output_triples}")
    if telemetry_json:
        typer.echo(f"🧭 Telemetry JSON -> {telemetry_json}")
    if telemetry_csv:
        typer.echo(f"🧭 Telemetry CSV  -> {telemetry_csv}")

if __name__ == "__main__":
    app()
