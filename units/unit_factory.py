################################################################################
#
#  Copyright (C) 2024 Sean Garrett-Roe
#  This file is part of omds - https//github.com/sgarrettroe/omds
#
#  This file is derived from pyqudt/ontology/unit_factory.py
#  Copyright (C) 2019 Garrett Brown
#  That file is part of pyqudt - https://github.com/eigendude/pyqudt
#
#  pyqudt is derived from jQUDT
#  Copyright (C) 2012-2013  Egon Willighagen <egonw@users.sf.net>
#
#  SPDX-License-Identifier: GPL-3.0-or-later
#  See the file LICENSE for more information.
#
################################################################################

import os

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import rdflib

from .unit import Unit

# set up logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-24s %(levelname)-8s %(message)s'
)
logger = logging.getLogger('omds.units.unit_factory')
logger.setLevel(logging.DEBUG)

# Type definitions
Statement = Tuple[str, str, rdflib.term.Identifier]
Predicate = Callable[[str, str, rdflib.term.Identifier], bool]


class UnitFactory:
    """
    A factory for creating units of measurement.
    """

    _instance: Optional['UnitFactory'] = None

    def __init__(self):
        """
        Create an instance of the unit factory.
        """
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

        self.g = g


    @classmethod
    def _get_instance(cls) -> 'UnitFactory':
        """
        Get the singleton used to store repository contents.

        :return: The singleton instance of type UnitFactory
        """
        if not cls._instance:
            cls._instance = UnitFactory()

        return cls._instance


    @classmethod
    def get_unit_by_name(cls, name: str) -> Unit:
        """
        Get a unit by its name as listed in http://qudt.org/vocab/units

        :param name: The unit's name as a string
        :return: The unit or None
        """
        # trying to get just one
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

        qres = cls._get_instance().g.query(qry)
        logger.debug(f"Found {len(qres)} query matches")
        for row in qres:
            logger.debug(
                f"unit:{name} mult {row.conversionMultiplier} offset {row.conversionOffset if row.conversionOffset else 0} label {row.label} symb {row.symbol} kind {row.quantityKind}")

        # resource_iri: str
        # label: str = dataclasses.field(default_factory=str)
        # abbreviation: str = dataclasses.field(default_factory=str)
        # symbol: str = dataclasses.field(default_factory=str)
        # type_iri: str = dataclasses.field(default_factory=str)
        # multiplier: Multiplier = dataclasses.field(default_factory=Multiplier)
        row = qres[0]
        return Unit(
            resource_iri='http://qudt.org/vocab/unit/' + name,
            symbol=row.symbol,
            label=row.label,
            multiplier=Multiplier(offset=row.conversionOffset,
                                  multiplier=row.conversionMultiplier),
            quantitykind_iri=row.quantityKind
            )


    @classmethod
    def get_unit(cls, resource_iri: str) -> Unit:
        """
        Get a unit by its resource IRI.

        :param resource_iri: The unit's resource IRI
        :return: The unit, or None on error
        """
        return cls._get_instance()._get_unit(resource_iri)

    @classmethod
    def get_qudt(cls, unit: str) -> Unit:
        """
        Shorthand for getting a QUDT unit like Joule (`J`) or meter-squared (`M2`)
        """
        return cls.get_unit('http://qudt.org/vocab/unit/' + unit)

    def _get_unit(self, resource_iri: str) -> Unit:
        """
        Internal implementation of get_unit().
        """
        unit = Unit(
            resource_iri=resource_iri,
        )

        statements: List[Statement] = self._get_statements(
            self._repos,
            lambda subj, pred, obj: str(subj) == resource_iri,
        )

        for (subject, predicate, obj) in statements:
            if predicate == QUDT.SYMBOL:
                unit.symbol = str(obj)
            elif predicate == QUDT.ABBREVIATION:
                unit.abbreviation = str(obj)
            elif predicate == QUDT.CONVERSION_OFFSET:
                unit.multiplier.offset = float(obj)
            elif predicate == QUDT.CONVERSION_MULTIPLIER:
                unit.multiplier.multiplier = float(obj)
            elif predicate == RDFS.LABEL:
                unit.label = str(obj)
            elif predicate == RDF.TYPE:
                type_iri = str(obj)
                if not self._should_be_ignored(type_iri):
                    unit.type_iri = type_iri

        return unit

    @classmethod
    def find_units(cls, abbreviation: str) -> List[Unit]:
        """
        Get units by their abbreviation.

        :param abbreviation: The unit abbreviation, e.g. 'nM'
        :return: The list of units, or empty if no units matched the abbreviation
        """
        return cls._get_instance()._find_units(abbreviation)

    def _find_units(self, abbreviation: str) -> List[Unit]:
        """
        Internal implementation of find_units()
        """
        found_units: List[Unit] = list()

        statements: List[Statement] = self._get_statements(
            self._repos,
            lambda subj, pred, o: str(pred) == QUDT.ABBREVIATION
            and str(o) == abbreviation,
        )

        for (subject, predicate, obj) in statements:
            type_iri = subject
            found_units.append(self._get_unit(type_iri))

        return found_units

    @classmethod
    def get_iris(cls, type_iri: str) -> List[str]:
        """
        Return a list of unit IRIs with the given unit type.

        :param type_iri: The IRI of the unit type, e.g. 'http://qudt.org/schema/qudt#TemperatureUnit'
        :return: The list of units, or empty if none match the specified type
        """
        return cls._get_instance()._get_iris(type_iri)

    def _get_iris(self, type_iri: str) -> List[str]:
        """
        Internal implementation of get_iris()
        """
        statements: List[Statement] = self._get_statements(
            self._repos,
            lambda subj, pred, o: str(o) == type_iri,
        )

        return [subj for (subj, pred, o) in statements]

    def _read_repo(self, file_name: str) -> rdflib.Graph:
        """
        Helper function to load the RDF triplet repository.

        :param file_name: The path to the repo
        :return: The loaded graph object
        """
        repo_path = os.path.join(self._repo_path, file_name)

        return OntologyReader.read(repo_path)

    @staticmethod
    def _get_statements(
        repos: List[rdflib.Graph], triplet_test: Predicate
    ) -> List[Statement]:
        """
        Get the statements of the given repos that satisfy the provided lambda.

        :param repos: The ontology repositories
        :param triplet_test: The lambda to invoke per statement
        :return: The matching statements
        """
        statements: List[Statement] = list()

        for repo in repos:
            for (subject, predicate, obj) in repo:
                if triplet_test(subject, predicate, obj):
                    statements.append((str(subject), str(predicate), obj))

        return statements

    @staticmethod
    def _should_be_ignored(type_iri: str) -> bool:
        """
        Check if a statement should be ignored when constructing units.

        :param type_iri: The predicate type
        :return: True if the statement should be ignored, False otherwise
        """
        # Accept anything outside the QUDT namespace
        if not type_iri.startswith(QUDT.namespace):
            return False

        if type_iri in [
            QUDT.SI_DERIVED_UNIT,
            QUDT.SI_BASE_UNIT,
            QUDT.SI_UNIT,
            QUDT.DERIVED_UNIT,
            QUDT.NOT_USED_WITH_SI_UNIT,
            QUDT.USED_WITH_SI_UNIT,
        ]:
            return True

        # Everything else is fine too
        return False
