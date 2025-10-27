# src/pipeline/validate_graph.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
import typer
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS, DCTERMS

MY = Namespace("https://w3id.org/my-project/ontology/")

app = typer.Typer(help="Run sanity SPARQL checks on the generated KG.")

def run_query(g: Graph, q: str):
    return list(g.query(q))

@app.command("run")
def run(
    triples: Path = typer.Option(..., "--triples", "-t", help="TTL produced by graph population"),
    ontology: Path = typer.Option("ontology/my_project_ontology.ttl", "--ontology", "-o", help="Project ontology TTL"),
    report: Path = typer.Option("data/telemetry/sparql_report.json", "--report", "-r", help="Where to write JSON report"),
    fail_on_zero_mapped: bool = typer.Option(True, help="Non-zero exit if no mapped properties are present"),
    fail_on_missing_provenance: bool = typer.Option(True, help="Non-zero exit if any instance lacks dcterms:source"),
):
    g = Graph()
    # Load ontology first (so labels/domains are available), then the data
    if Path(ontology).exists():
        g.parse(str(ontology), format="turtle")
    g.parse(str(triples), format="turtle")

    # Q1: Count mapped edges (exclude my:relatedToRaw)
    q1 = """
    PREFIX my: <https://w3id.org/my-project/ontology/>
    SELECT (COUNT(*) AS ?mappedEdgeCount)
    WHERE {
      ?s ?p ?o .
      FILTER (?p != my:relatedToRaw)
    }
    """
    mapped_count = int(run_query(g, q1)[0][0])

    # Q2: Instances lacking provenance (dcterms:source) — exclude RawRelation bnodes
    q2 = """
    PREFIX my: <https://w3id.org/my-project/ontology/>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    SELECT (COUNT(DISTINCT ?s) AS ?noProv)
    WHERE {
      ?s a ?t .
      FILTER (?t != my:RawRelation)
      FILTER NOT EXISTS { ?s dcterms:source ?src }
    }
    """
    missing_prov = int(run_query(g, q2)[0][0])

    # Q3: Top properties by frequency (simple histogram)
    q3 = """
    SELECT ?p (COUNT(*) AS ?c)
    WHERE { ?s ?p ?o }
    GROUP BY ?p ORDER BY DESC(?c) LIMIT 20
    """
    top_props = [{"property": str(row[0]), "count": int(row[1])} for row in run_query(g, q3)]

    # Q4: Sample of mapped custom edges (assesses/exhibitsBehavior/spreads)
    q4 = """
    PREFIX my: <https://w3id.org/my-project/ontology/>
    SELECT ?p ?s ?o
    WHERE {
      VALUES ?p { my:assesses my:exhibitsBehavior my:spreads }
      ?s ?p ?o
    } LIMIT 10
    """
    samples = [{"p": str(r[0]), "s": str(r[1]), "o": str(r[2])} for r in run_query(g, q4)]

    report_data: Dict[str, Any] = {
        "mapped_edges": mapped_count,
        "instances_missing_provenance": missing_prov,
        "top_properties": top_props,
        "custom_property_samples": samples,
    }
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # Console summary
    typer.echo(json.dumps(report_data, indent=2))

    # Guardrails
    exit_code = 0
    if fail_on_zero_mapped and mapped_count == 0:
        typer.echo("❌ No mapped edges found (all relations may be falling back to my:relatedToRaw).", err=True)
        exit_code = 1
    if fail_on_missing_provenance and missing_prov > 0:
        typer.echo(f"❌ {missing_prov} instances missing dcterms:source.", err=True)
        exit_code = 1

    raise typer.Exit(code=exit_code)

if __name__ == "__main__":
    app()
