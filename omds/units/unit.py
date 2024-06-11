################################################################################
#
#  Copyright (C) 2024 Sean Garrett-Roe
#  This file is part of omds - https//github.com/sgarrettroe/omds
#
#  This file is derived from pyqudt/units/unit.py
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

from .multiplier import Multiplier


@dataclasses.dataclass
class Unit:
    """
    A unit of measurement.
    """

    resource_iri: str
    label: str = dataclasses.field(default_factory=str)
    symbol: str = dataclasses.field(default_factory=str)
    quantitykind_iri: str = dataclasses.field(default_factory=str)
    multiplier: Multiplier = dataclasses.field(default_factory=Multiplier)

    def __repr__(self) -> str:
        return str(self.label)
