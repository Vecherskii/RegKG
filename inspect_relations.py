import rdflib

g = rdflib.Graph()
g.parse(r"data\processed\knowledge_graph_triples_v4.ttl", format="turtle")

q = """
PREFIX my: <https://w3id.org/my-project/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?p ?sLabel ?oLabel
WHERE {
  VALUES ?p { my:assesses my:spreads my:exhibitsBehavior }
  ?s ?p ?o .
  ?s rdfs:label ?sLabel .
  ?o rdfs:label ?oLabel .
}
"""

for row in g.query(q):
    print(row)