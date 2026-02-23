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
    # Registry maps machine names to transpiler classes (not instances)
    __registry = {
        "pfaff_v1": XYiSW,
    }

    # Lazily-populated instance cache
    __instances = {}

    def __init__(self):
        # Reserved for future stateful use
        pass

    def get(self, mach: str):
        """Get the specific transpiler for the right architecture.

        :param mach: name of the architecture.
        :return: the transpiler object.
        """
        if mach not in self.__instances:
            if mach not in self.__registry:
                raise KeyError(f"Unknown machine '{mach}'. Available: {list(self.__registry.keys())}")
            self.__instances[mach] = self.__registry[mach]()

        return self.__instances[mach]
