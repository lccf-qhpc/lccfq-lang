"""
Filename: mach.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file contains factories that arise from the machine (device) model

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from ...mach.sets.xyisqswap import XYiSW


class TranspilerFactory:
    """A transpiler factory that selects a specific transpiler based on
    specification of a machine.
    """
    # Internal set of transpiler choices
    __transpilers = {
        "pfaff_v1": XYiSW
    }

    def __init__(self):
        # Reserved for future stateful use
        pass

    def get(self, mach: str):
        """Get the specific transpiler for the right architecture.

        :param mach: name of the architecture.
        :return: the transpiler object.
        """
        return self.__transpilers[mach]
