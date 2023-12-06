import numpy as np
import rdflib
import pprint

QUDT = rdflib.Namespace("https://qudt.org/2.1/vocab/unit")
g = rdflib.Graph()
g.bind("qudt",QUDT)
#units = rdflib.URIRef("https://qudt.org/2.1/vocab/unit")
#print(units)
g.parse("https://qudt.org/2.1/vocab/unit")
#print(g.serialize(format="turtle"))
print(f"found {len(g)} units")

for unit in g.subjects():
    pprint.pprint(unit)
#
# I can see the unit names in URIRef(...) but I don't know how to use them yet. I want to be able to get the units
# and their conversion factors automagically
#
# qudt:Quantity -> qudt:QuantityValue, qudt:QuantityKind
