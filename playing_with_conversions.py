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
OMDS = rdflib.Namespace("http://www.semanticweb.org/sgr/ontologies/2024/1/omds/")
g = rdflib.Graph(bind_namespaces="rdflib")
g.bind("qudt:", QUDT)
g.bind("unit:", UNIT)
g.bind("quantitykind:", QUANTITY_KIND)
g.bind("constant:", CONSTANT)
g.bind(":", OMDS)

#
# back to conversions
#
g.parse("omds.ttl")
g.parse('http://qudt.org/vocab/constant')
g.parse('http://qudt.org/vocab/unit')


# for s in g.subjects():
#     print(f"found subject {s}")
# print('-'*76)

qk_from = "quantitykind:Length"
qk_to = "quantitykind:Frequency"
#qk_to = "quantitykind:Energy"
qry = f"""
    SELECT ?r ?expr ?expression_input_name ?rr ?ern ?erv #?ervv
    WHERE {{
    #?r :recipeConvertsTo ?qk .
    ?r :recipeConvertsTo {qk_to} .
    ?r :recipeConvertsFrom {qk_from} .
    ?r :expression ?expr .
    ?r :expressionInputName ?expression_input_name .
    ?r :recipeReplacement ?rr .
    ?rr :expressionReplacementName ?ern .
    ?rr :expressionReplacementValue ?erv .
    #?erv qudt:value ?ervv .
    }}"""
qres = g.query(qry)

ld = {}
for row in qres:
    print(f'{row.r.n3(namespace_manager=g.namespace_manager)}: ')
    print(f'\t{row.expr}')
    print(f'\tein:{row.expression_input_name}')
    print(f'\trr:{row.rr.n3(namespace_manager=g.namespace_manager)}')
    #ervv = constant_dict[row.erv.n3(namespace_manager=g.namespace_manager)]
    for val in g.objects(row.erv, QUDT.value):
        ervv = val.toPython()
    print(f'\tern:{row.ern} erv:{row.erv.n3(namespace_manager=g.namespace_manager)} ervv:{ervv}')
    ld.update({row.ern.toPython():ervv})

print(ld)
print("done")