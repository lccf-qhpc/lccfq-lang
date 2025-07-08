"""
Filename: transpilers.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides transpilation for devices using X, Y and sqrt(iSWAP) gates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from abc import ABC, abstractmethod
from typing import List
from .ir import Gate
from ..arch.instruction import Instruction


class Transpiler(ABC):
    """
    A transpiler is an object that uses a mapping and a specific rendition of
    an instruction to produce native gates for a specific QPU.
    """

    def __init__(self):
        self.mapping = False
        self.topology = None

    @abstractmethod
    def transpile(self, instruction: Instruction) -> List[Gate]:
        pass


class XYiSW(Transpiler):
    """Transpilation class for Pfaff Lab hardware.
    """

    def transpile(self, instruction: Instruction) -> List[Gate]:
        """Transpile an instruction into a sequence of gates. The result is a list
        since gate ordering matters. The process resembles a dispatch.

        :param instruction: The instruction to transpile.
        :return: A list of gates implementing that instruction.
        """
        dispatch = {
            "id": self._id,
        }

        return dispatch[instruction.symbol](instruction)

    @staticmethod
    def _id(instruction: Instruction):
        return []

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
