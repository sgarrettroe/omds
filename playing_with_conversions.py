# playing around with my version of units
import numbers
import rdflib
from pprint import pprint
from units.unit import Unit
from units.multiplier import Multiplier
from units.quantity import Quantity
import re
from keyword import iskeyword
import math
import logging

# for debugging / testing (can/should be removed)
import os
from pprint import pprint

# set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-24s %(levelname)-8s %(message)s'
)
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

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

qk_from = "quantitykind:Length"
qk_to = "quantitykind:Frequency"
#qk_to = "quantitykind:Energy"
qry = f"""
    SELECT ?r ?expr ?expression_input_name ?rr ?ern ?erv #?ervv
    WHERE {{
    #?r :relationConvertsTo ?qk .
    ?r :relationConvertsTo {qk_to} .
    ?r :relationConvertsFrom {qk_from} .
    ?r :expression ?expr .
    ?r :expressionInputName ?expression_input_name .
    ?r :relationReplacement ?rr .
    ?rr :expressionReplacementName ?ern .
    ?rr :expressionReplacementValue ?erv .
    #?erv qudt:value ?ervv .
    }}"""
qres = g.query(qry)

ld = {}
for row in qres:
    express = row.expr.toPython()
    ein = row.expression_input_name.toPython()
    logger.debug(f'{row.r.n3(namespace_manager=g.namespace_manager)}: ')
    logger.debug(f'\t{row.expr}')
    logger.debug(f'\tein:{row.expression_input_name}')
    logger.debug(f'\trr:{row.rr.n3(namespace_manager=g.namespace_manager)}')
    #ervv = constant_dict[row.erv.n3(namespace_manager=g.namespace_manager)]
    for val in g.objects(row.erv, QUDT.value):
        ervv = val.toPython()
    logger.debug(f'\tern:{row.ern} erv:{row.erv.n3(namespace_manager=g.namespace_manager)} ervv:{ervv}')
    ld.update({row.ern.toPython():ervv})


def is_valid_identifier(name: str)->bool:
    """Test identifiers to conform to Python rules EXCEPT
    do NOT allow the first character to be _ (underscore).
    """
    m = re.fullmatch('^[a-zA-Z]\w*',name)
    return m is not None and not iskeyword(name)


def execute(expression:str, substitutions=None):
    """Execute expression applying substitution dictionary.

    Ideas to mitigate security vulnerabilities were inspired by
    https://realpython.com/python-eval-function/
    """
    # default substitutions
    if substitutions is None:
        substitutions = {}

    # only allow math operators and functions REMOVING all that start _ and __
    ALLOWED_NAMES = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }

    # make sure substitutions are plugging in numbers only
    for k, v in substitutions.items():
        if is_valid_identifier(k):
            pass
        else:
            raise NameError(f"The use of '{k}' is not allowed")
        if ((isinstance(v,str) and v.isnumeric)
                or (isinstance(v,numbers.Number))):
            pass
        else:
            raise ValueError(f"The value of '{k}': '{v}' is not allowed")
    ALLOWED_NAMES.update(substitutions)

    # Compile the expression
    code = compile(expression, "<string>", "eval")

    # Validate allowed names
    for name in code.co_names:
        if name not in ALLOWED_NAMES:
            raise NameError(f"The use of '{name}' is not allowed")

    logger.debug(f'Expression to evaluate:\n{expression}')
    logger.debug(f'with variable substitutions\n{substitutions}')

    return eval(code, {"__builtins__": {}}, ALLOWED_NAMES)


# some things for testing
tryme = ['hello',  # true
         'hello_world',  # true
         '_hello',  # false
         '__hello',  # false
         'hello.world',  # false
         'hello(world',  # false
         'if',  # false
         'while',  # false
         ]

for n in tryme:
    print(is_valid_identifier(n))


val = 632.8e-9
ld.update({ein:val})
ret = execute(express,ld)
print(ret)
print(1/ret)
print("done")

