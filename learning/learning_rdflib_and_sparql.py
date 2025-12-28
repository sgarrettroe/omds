import numpy as np
import rdflib
from pprint import pprint

# not working
# g.parse("http://danbri.org/foaf.rdf#")
# g.parse("https://goldbook.iupac.org/")


QUDT = rdflib.Namespace("http://qudt.org/schema/qudt/")
UNIT = rdflib.Namespace("https://qudt.org/2.1/vocab/unit")
SOU = rdflib.Namespace("http://qudt.org/vocab/sou/")

g = rdflib.Graph()
g.bind("qudt", QUDT)
g.bind("sou", SOU)
g.bind("unit", UNIT)

# units = rdflib.URIRef("https://qudt.org/2.1/vocab/unit")
# print(units)
g.parse("https://qudt.org/2.1/vocab/unit")
# print(g.serialize(format="turtle"))

print(f"found {len(g)} units")

# for unit in g.subjects():
#     pprint(unit)
#
# I can see the unit names in URIRef(...) but I don't know how to use them yet. I want to be able to get the units
# and their conversion factors automagically
#
# qudt:Quantity -> qudt:QuantityValue, qudt:QuantityKind


qry = """
PREFIX unit: <http://qudt.org/vocab/unit/> 
PREFIX qudt: <http://qudt.org/schema/qudt/>
PREFIX sou: <http://qudt.org/vocab/sou/>
PREFIX quantitykind: <http://qudt.org/vocab/quantitykind/> 

SELECT ?x ?y
WHERE {
    ?x qudt:applicableSystem sou:SI .
    ?x qudt:conversionMultiplier ?y .
    ?x qudt:hasQuantityKind quantitykind:Time .
}"""
qres = g.query(qry)
pprint(f"Found {len(qres)} query matches")
for row in qres:
    pprint(f"{row.x} val {row.y}")

# trying to get just one
name = 'SEC'
qry = """
PREFIX unit: <http://qudt.org/vocab/unit/> 
PREFIX qudt: <http://qudt.org/schema/qudt/>
PREFIX sou: <http://qudt.org/vocab/sou/>
PREFIX quantitykind: <http://qudt.org/vocab/quantitykind/> 

SELECT ?conversionMultiplier ?symbol ?label ?
WHERE {
    unit:SEC qudt:conversionMultiplier ?conversionMultiplier .
    unit:SEC qudt:symbol ?symbol .
}"""

qres = g.query(qry)
pprint(f"Found {len(qres)} query matches")
for row in qres:
    pprint(f"unit:SEC val {row.conversionMultiplier}")

# no atto or femto seconds!!!!