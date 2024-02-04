################################################################################
#
#  Copyright (C) 2024 Sean Garrett-Roe
#  This file is part of omds - https//github.com/sgarrettroe/omds
#
#  This file is derived from pyqudt/units/multiplier.py
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


@dataclasses.dataclass
class Multiplier:
    """
    A multiplier with an optional offset.
    """

    offset: float = dataclasses.field(default=0.0)
    multiplier: float = dataclasses.field(default=1.0)
