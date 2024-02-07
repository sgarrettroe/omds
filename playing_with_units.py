# playing around with my version of units
import rdflib
from pprint import pprint
from units.unit import Unit
from units.multiplier import Multiplier
from units.quantity import Quantity

QUDT = rdflib.Namespace("http://qudt.org/schema/qudt/")
UNIT = rdflib.Namespace("https://qudt.org/vocab/unit/")
QUANTITY_KIND = rdflib.Namespace("https://qudt.org/vocab/quantitykind/")
CONSTANT = rdflib.Namespace("https://qudt.org/vocab/constant/")
namespaces = {'qudt': QUDT,
              'unit': UNIT,
              'quantitykind': QUANTITY_KIND,
              'constant': CONSTANT}
g = rdflib.Graph(bind_namespaces="rdflib")
g.bind("qudt:", QUDT)
g.bind("unit:", UNIT)
g.bind("quantitykind:", QUANTITY_KIND)
g.bind("constant:", CONSTANT)

g.parse("https://qudt.org/2.1/vocab/unit")

# trying to get some
l = ['SEC', 'PicoSEC', 'HZ', 'TeraHZ', 'NanoM']
for name in l:
    qry = f"""
   
    SELECT ?conversionMultiplier ?conversionOffset ?symbol ?quantityKind ?label 
    WHERE {{
        unit:{name} qudt:conversionMultiplier ?conversionMultiplier .
        unit:{name} qudt:symbol ?symbol .
        unit:{name} qudt:hasQuantityKind ?quantityKind .
        unit:{name} rdfs:label ?label .
        OPTIONAL {{ unit:{name} qudt:conversionOffset ?conversionOffset }}
        FILTER (lang(?label) = 'en')
        FILTER (?quantityKind = quantitykind:Time || ?quantityKind = quantitykind:Frequency || ?quantityKind = quantitykind:Length)
    }}"""

    qres = g.query(qry)
    pprint(f"Found {len(qres)} query matches")

    for row in qres:
        pprint(f"unit:{name} mult {row.conversionMultiplier} \
        offset { row.conversionOffset if row.conversionOffset else 0} \
        label {row.label} symb {row.symbol} kind {row.quantityKind}")

        pprint(namespaces['unit'][name].n3())
        pprint(namespaces['quantitykind'][row.quantityKind].n3())

        #d[name] = Unit(f'{UNIT/{row.name}',label=row.label,symbol=row.symbol,quantitykind_iri=f'{quantitykind}/{row.quantityKind}')

QK_FILTER = """FILTER (
        ?quantityKind = quantitykind:Action || 
        ?quantityKind = quantitykind:Dimensionless || 
        ?quantityKind = quantitykind:Energy || 
        ?quantityKind = quantitykind:Frequency || 
        ?quantityKind = quantitykind:InverseLength ||
        ?quantityKind = quantitykind:Length || 
        ?quantityKind = quantitykind:Mass || 
        ?quantityKind = quantitykind:MolarMass ||
        ?quantityKind = quantitykind:Speed ||
        ?quantityKind = quantitykind:Time || 
        ?quantityKind = quantitykind:Temperature
        )"""
qry = f"""
    SELECT ?name ?conversionMultiplier ?conversionOffset ?symbol ?quantityKind ?label 
    WHERE {{
        ?name qudt:conversionMultiplier ?conversionMultiplier .
        ?name qudt:symbol ?symbol .
        ?name qudt:hasQuantityKind ?quantityKind .
        ?name rdfs:label ?label .
        OPTIONAL {{ ?name qudt:conversionOffset ?conversionOffset }}
        FILTER (lang(?label) = 'en')
        {QK_FILTER}
    }}"""
unit_qry = qry
qres = g.query(qry)

pprint(len(qres))
unit_dict = {}
for row in qres:
    n = row.name.n3(namespace_manager=g.namespace_manager)
    #cm = row.conversionMultiplier.n3(namespace_manager=g.namespace_manager)
    cm = row.conversionMultiplier.toPython()
    co = row.conversionOffset.n3(namespace_manager=g.namespace_manager) if row.conversionOffset else 0
    co = row.conversionOffset.toPython() if row.conversionOffset else 0
    s = row.symbol.n3(namespace_manager=g.namespace_manager)
    qk = row.quantityKind.n3(namespace_manager=g.namespace_manager)
    l = row.label.n3(namespace_manager=g.namespace_manager)
    r_iri = row.name.n3()
    qk_iri = row.quantityKind.n3()
    unit_dict[n] = Unit(resource_iri=r_iri,
                           label=l,
                           symbol=s,
                           quantitykind_iri=qk_iri,
                           multiplier=Multiplier(offset=co,multiplier=cm))

pprint(len(unit_dict))

pprint(unit_dict.keys())
UNIT_FILTER = ['FILTER (\n']
for k in unit_dict.keys():
    UNIT_FILTER.append(f'        ?u unit:kind {k} ||\n')
UNIT_FILTER[-1].rstrip('||\n')
UNIT_FILTER.append(')\n')
UNIT_FILTER = ''.join(UNIT_FILTER)

#
# Trying getting constants
#
g.parse("https://qudt.org/2.1/vocab/constant")

qry = f"""
    SELECT ?constant ?v ?u
    WHERE {{
        ?constant a qudt:PhysicalConstant .
        ?constant qudt:hasQuantityKind ?quantityKind .
        ?constant qudt:quantityValue ?val .
        ?val qudt:hasUnit ?u .
        ?u qudt:applicableSystem sou:SI .
        ?val qudt:value ?v .
        ?val qudt:hasUnit ?name
        # use a subquery to select constants in the set of units we selected
        {{
        {unit_qry}
        }}
    }}
"""
#        {QK_FILTER}

print(qry)
qres = g.query(qry)
pprint(len(qres))
constant_dict = {}
for row in qres:
    n = row.constant.n3(namespace_manager=g.namespace_manager)
    v = row.v.toPython()
    u = row.u.n3(namespace_manager=g.namespace_manager)
    pprint(f'{n} {v} {u}')
    constant_dict[n] = Quantity(v, unit_dict[u])

pprint(unit_dict)
pprint(constant_dict)
print("done")