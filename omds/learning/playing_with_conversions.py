# playing around with my version of units
import numbers
import rdflib
from pprint import pprint
from omds.units.unit import Unit
from omds.units.unit_factory import UnitFactory
from omds.units.multiplier import Multiplier
from omds.units.quantity import Quantity
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
    import math
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


#
# back to conversions
#

def use_relation_to_convert(qty_in: Quantity, unit_out:Unit)->Quantity:

    val_in = qty_in.value
    QUDT = rdflib.Namespace("http://qudt.org/schema/qudt/")

    # TODO: use multiplier to switch to base units of input
    # qty_in.convert_to('base')

    qk_from = qty_in.unit.quantitykind_iri
    qk_to = unit_out.quantitykind_iri
    if qk_from != qk_to:
        logger.debug(f'found differing quantity kinds:\nfrom:{qk_from} to:{qk_to}. \
        Attempting to find a physical relationship to convert them.')
    # Use SPARQL to try to find a relation that connects the two quantities
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
    qres = UnitFactory.query(qry)

    # extract results to find the relation and relation replacements
    # (i.e., physical constants and the input variable name)
    ld = {}
    for row in qres:
        r = row.r.n3(namespace_manager=UnitFactory._get_instance().g.namespace_manager)
        rr = row.rr.n3(namespace_manager=UnitFactory._get_instance().g.namespace_manager)
        express = row.expr.toPython()
        ein = row.expression_input_name.toPython()
        logger.debug(f'{r}: ')
        logger.debug(f'\t{express}')
        logger.debug(f'\tein:{ein}')
        logger.debug(f'\trr:{rr}')

        for val in UnitFactory._objects(row.erv, QUDT.value):
            ervv = val.toPython()
        logger.debug(f'\tern:{row.ern} erv:{row.erv.n3(namespace_manager=UnitFactory._get_instance().g.namespace_manager)} ervv:{ervv}')
        ld.update({row.ern.toPython(): ervv})

    # add the input value to the dictionary of replacements
    ld.update({ein: val_in})

    # run the expression
    ret = execute(express,ld)

    # TODO: Use multiplier to switch from base units in output
    # qty_out = Quantity(ret, UnitFactory.get_unit(Quantity._base_unit[qk_out]))
    # qty_out = qty_out.convert_to(unit_out)

    qty_out = Quantity(ret,unit_out)
    return qty_out


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

u1 = UnitFactory.get_unit_by_name('unit:M')
u2 = UnitFactory.get_unit_by_name('unit:PER-SEC')
q1 = Quantity(632.8e-9, u1)
q2 = use_relation_to_convert(q1, u2)
print(q2)
print(1/q2.value)

q3 = Quantity(632.8, UnitFactory.get_unit_by_name('unit:NanoM'))
print(q3)
q4 = q3.convert_to(UnitFactory.get_unit_by_name('unit:M'))
print(q4)

q5 = q3.convert_to(UnitFactory.get_unit_by_name('unit:PER-SEC'))
print(q5)
print("done")

