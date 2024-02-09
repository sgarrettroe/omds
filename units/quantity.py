################################################################################
#
#  Copyright (C) 2024 Sean Garrett-Roe
#  This file is part of omds - https//github.com/sgarrettroe/omds
#
#  This file is derived from pyqudt/units/quantity.py
#  Copyright (C) 2019 Garrett Brown
#  That file is part of pyqudt - https://github.com/eigendude/pyqudt
#
#  pyqudt is derived from jQUDT
#  Copyright (C) 2012-2013  Egon Willighagen <egonw@users.sf.net>
#
#  SPDX-License-Identifier: BSD-3-Clause
#  See the file LICENSE for more information.
#
################################################################################

import dataclasses
import rdflib
from typing import Optional
from .unit import Unit
from .unit_factory import UnitFactory
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

@dataclasses.dataclass
class Quantity(object):
    """
    A quantity with a value and a unit.
    """

    value: float
    unit: Optional[Unit]

    _base_units = {'quantitykind:AmountOfSubstance': 'unit:MOL',
                   'quantitykind:LuminousIntensity': 'unit:CD',
                   'quantitykind:Mass': 'unit:KiloGM',
                   'quantitykind:Length': 'unit:M',
                   'quantitykind:Time': 'unit:SEC',
                   'quantitykind:ElectricCurrent': 'unit:A',
                   'quantitykind:Temperature': 'unit:K',
                   'quantitykind:Frequency': 'PER-SEC',
                   'quantitykind:InverseTime': 'PER-M'
                   }
    def convert_to(self, unit: Unit | str) -> 'Quantity':
        """
        Converts the quantity's value to the specified unit of measurement.

        :param unit: The target unit or 'base'
        :return: The converted quantity
        """

        if not unit:
            raise ValueError('Target unit cannot be null, must be a unit or "base".')

        if not self.unit:
            raise ValueError('This measurement does not have units defined')

        if self.unit == unit:
            # Nothing to be done
            return self

        qk_from = self.unit.quantitykind_iri
        qk_to = unit.quantitykind_iri
        base_flag = False
        if isinstance(unit, str):
            logger.warning(
                '!!!This is untested and probably needs to be fixed!!!'
            )
            if unit == 'base':
                base_flag = True
                unit = UnitFactory.get_unit(self._base_units[qk_from])
            else:
                unit = UnitFactory.get_unit(unit)


        # Convert to the base unit
        base_unit_value = (
            float(self.value) * float(self.unit.multiplier.multiplier) + float(self.unit.multiplier.offset)
        )

        if qk_from != qk_to:
            base_unit_value = Quantity.use_relation_to_convert(base_unit_value, qk_from, qk_to)

        # Convert the base unit to the new unit
        if base_flag:
            new_value = base_unit_value
        else:
            new_value = (
                float(base_unit_value) - float(unit.multiplier.offset)
            ) / float(unit.multiplier.multiplier)

        new_measurement = Quantity(
            unit=unit,
            value=new_value,
        )

        return new_measurement

    @classmethod
    def use_relation_to_convert(cls, val_in, qk_from, qk_to) -> float:

        def is_valid_identifier(name: str) -> bool:
            """Test identifiers to conform to Python rules EXCEPT
            do NOT allow the first character to be _ (underscore).
            """
            import re
            from keyword import iskeyword
            m = re.fullmatch('^[a-zA-Z]\w*', name)
            return m is not None and not iskeyword(name)

        def execute(expression: str, substitutions=None):
            """Execute expression applying substitution dictionary.

            Ideas to mitigate security vulnerabilities were inspired by
            https://realpython.com/python-eval-function/
            """
            import math
            import numbers

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
                if ((isinstance(v, str) and v.isnumeric)
                        or (isinstance(v, numbers.Number))):
                    pass
                else:
                    raise ValueError(
                        f"The value of '{k}': '{v}' is not allowed")
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

        QUDT = rdflib.Namespace("http://qudt.org/schema/qudt/")

        if qk_from != qk_to:
            logger.debug(
                f'found differing quantity kinds:\n\tfrom:{qk_from} to:{qk_to}. \
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
            r = row.r.n3(
                namespace_manager=UnitFactory._get_instance().g.namespace_manager)
            rr = row.rr.n3(
                namespace_manager=UnitFactory._get_instance().g.namespace_manager)
            express = row.expr.toPython()
            ein = row.expression_input_name.toPython()
            logger.debug(f'{r}: ')
            logger.debug(f'\t{express}')
            logger.debug(f'\tein:{ein}')
            logger.debug(f'\trr:{rr}')

            for val in UnitFactory._objects(row.erv, QUDT.value):
                ervv = val.toPython()
            logger.debug(
                f'\tern:{row.ern} erv:{row.erv.n3(namespace_manager=UnitFactory._get_instance().g.namespace_manager)} ervv:{ervv}')
            ld.update({row.ern.toPython(): ervv})

        # add the input value to the dictionary of replacements
        ld.update({ein: val_in})

        # run the expression
        ret = execute(express, ld)

        return ret

    def __repr__(self) -> str:
        """
        Return a string representation of the quantity.
        """
        return f'{self.value} {self.unit}'
