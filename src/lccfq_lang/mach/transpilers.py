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

from numpy import pi as PI
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
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
        self.table = None

    @abstractmethod
    def transpile(self, instruction: Instruction) -> List[Gate]:
        pass


class XYiSW(Transpiler):
    """Transpilation class for Pfaff Lab hardware.
    """

    _table = {
        "nop": [("nop", [])],
        "x": [("rx", [PI])],
        "y": [("ry", [PI])],
        "z": [("ry", [-PI / 2]), ("rx", [PI]), ("ry", [PI / 2])],
        "h": [("ry", [PI / 2]), ("rx", [PI])],
        "s": [("ry", [-PI / 2]), ("rx", [PI / 2]), ("ry", [PI / 2])],
        "sdg": [("ry", [-PI / 2]), ("rx", [-PI / 2]), ("ry", [PI / 2])],
        "t": [("ry", [-PI / 2]), ("rx", [PI / 4]), ("ry", [PI / 2])],
        "tdg": [("ry", [-PI / 2]), ("rx", [-PI / 4]), ("ry", [PI / 2])],
        "p": [("ry", [-PI / 2]), ("rx", None), ("ry", [PI / 2])],
        "rx": [("rx", None)],
        "ry": [("ry", None)],
        "rz": [("ry", [-PI / 2]), ("rx", None), ("ry", [PI / 2])],
        "phase": [("ry", [-PI / 2]), ("rx", None), ("ry", [PI / 2])],
        # Special case 1: u2 - must be decomposed at the instruction level into rz.ry.rz
        # Special case 2: u3 - must be decomposed at the instruction level into rz.ry.rz

        # dispatch = {
        # "cx": self._cx,
        # "cy": self._cy,
        # "cz": self._cz,
        # "ch": self._ch,
        # "cp": self._cp,
        # "crx": self._crx,
        # "cry": self._cry,
        # "crz": self._crz,
        # "cphase": self._cphase,
        # "cu": self._cu
        # }
    }

    def __init__(self):
        """Add the main initialization table that will drive the transpilation process.

        A value of `None` in the table indicates that parameters from the instruction
        should be used instead.
        """
        super().__init__()

    def transpile(self, instruction: Instruction) -> List[Gate]:
        """Transpile an instruction into a sequence of gates. The result is a list
        since gate ordering matters. The process resembles a dispatch. We use already
        mapped (and swapped) qubits.

        :param instruction: The instruction to transpile.
        :return: A list of gates implementing that instruction.
        """
        gate_maker = self._synthesize(instruction)
        return list(map(lambda g: gate_maker(*g), self._table[instruction.symbol]))

    @staticmethod
    def _synthesize(instruction: Instruction) -> Callable[[str, Optional[List[float]]], Gate]:
        """Synthesis method that produces a function which, with the right parameters, yields a gate.

        :param instruction: instruction used to synthesize one gate in the corresponding sequence
        :return: curried function that, upon parameters, completes the gates
        """
        def gate(symbol: str, params: Optional[List[float]]=None) -> Gate:
            return Gate(
                symbol=symbol,
                target_qubits=instruction.target_qubits,
                control_qubits=instruction.control_qubits,
                params=params if params is not None else instruction.parameters,
            )

        return gate

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
