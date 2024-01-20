# playing around with my version of units
import rdflib
from pprint import pprint
#from units.unit_factory import UnitFactory, Unit
#SEC: Unit = UnitFactory._get_instance().get_unit_by_name('SEC')

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

# trying to get just one
l = ['SEC', 'PicoSEC', 'HZ', 'TeraHZ', 'NanoM']
for name in l:
    qry = f"""
    PREFIX unit: <http://qudt.org/vocab/unit/> 
    PREFIX qudt: <http://qudt.org/schema/qudt/>
    PREFIX sou: <http://qudt.org/vocab/sou/>
    PREFIX quantitykind: <http://qudt.org/vocab/quantitykind/> 
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
    
    SELECT ?conversionMultiplier ?conversionOffset ?symbol ?quantityKind ?label 
    WHERE {{
        unit:{name} qudt:conversionMultiplier ?conversionMultiplier .
        unit:{name} qudt:symbol ?symbol .
        unit:{name} qudt:hasQuantityKind ?quantityKind .
        unit:{name} rdfs:label ?label .
        OPTIONAL {{ unit:{name} qudt:conversionOffset ?conversionOffset }}
        FILTER (lang(?label) = 'en')
    }}"""

    qres = g.query(qry)
    pprint(f"Found {len(qres)} query matches")
    for row in qres:
        pprint(f"unit:{name} mult {row.conversionMultiplier} offset { row.conversionOffset if row.conversionOffset else 0} label {row.label} symb {row.symbol} kind {row.quantityKind}")



