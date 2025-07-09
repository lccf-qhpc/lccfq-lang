"""
Filename: xyisqswap.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides transpilation for devices using X, Y and sqrt(iSWAP) gates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from numpy import pi as PI
from typing import List, Callable, Optional
from ..ir import Gate, Control, Test
from ..transpilers import Transpiler
from ...arch.instruction import Instruction


class XYiSW(Transpiler):
    """Transpilation class for Pfaff Lab hardware.
    """

    _table = {
        "nop": [("nop", [], ".")],
        "x": [("rx", [PI], ".")],
        "y": [("ry", [PI], ".")],
        "z": [
            ("ry", [-PI/2], "."),
            ("rx", [PI], "."),
            ("ry", [PI/2], ".")
        ],
        "h": [
            ("ry", [PI/2], "."),
            ("rx", [PI], ".")
        ],
        "s": [
            ("ry", [-PI/2], "."),
            ("rx", [PI/2], "."),
            ("ry", [PI/2], ".")
        ],
        "sdg": [
            ("ry", [-PI/2], "."),
            ("rx", [-PI/2], "."),
            ("ry", [PI/2], ".")
        ],
        "t": [
            ("ry", [-PI/2], "."),
            ("rx", [PI/4], "."),
            ("ry", [PI/2], ".")
        ],
        "tdg": [
            ("ry", [-PI/2], "."),
            ("rx", [-PI / 4], "."),
            ("ry", [PI/2], ".")
        ],
        "p": [
            ("ry", [-PI/2], "."),
            ("rx", None, "."),
            ("ry", [PI/2], ".")
        ],
        "rx": [("rx", None, ".")],
        "ry": [("ry", None, ".")],
        "rz": [
            ("ry", [-PI/2], "."),
            ("rx", None, "."),
            ("ry", [PI/2], ".")
        ],
        "phase": [
            ("ry", [-PI/2], "."),
            ("rx", None, "."),
            ("ry", [PI/2], ".")
        ],
        # Special case 1: u2 - must be decomposed at the instruction level into rz.ry.rz
        # Special case 2: u3 - must be decomposed at the instruction level into rz.ry.rz
        "swap" : [
            # We compile directly using sqiswaps to avoid 6 extra gates through the usual intermediate CNOTs
            # Other options to be considered in the future; see https://arxiv.org/html/2412.15022v1#bib.bib29
            ("rx", [PI/2], "c"),
            ("ry", [PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [PI/2], "c"),
            ("ry", [PI/2], "t"),
            ("sqiswap", [], "*")
        ],
        "cx": [
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t")
        ],
        "cy": [
            ("rx", [-PI/2], "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("rx", [PI/2], "t")
        ],
        "cz": [
            ("ry", [PI/2], "t"),
            ("rx", [PI], "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("ry", [PI/2], "t"),
            ("rx", [PI], "t")
        ],
        "ch": [
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t")
        ],
        "cp": [
            ("ry", [PI/2], "t"),
            ("rx", [PI/2], "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("rx", None, "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("rx", [-PI/2], "t"),
            ("ry", [-PI/2], "t"),
        ],
        "crx": [
            ("ry", [PI/2], "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("rx", None, "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("ry", [-PI/2], "t")
        ],
        "cry": [
            ("rx", [PI], "t"),
            ("ry", None, "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("ry", None, "t"),
            ("rx", [PI], "t")
        ],
        "crz": [
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("ry", [-PI/2], "t"),
            ("rx", None, "t"),
            ("ry", [PI/2], "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
        ],
        "cphase": [
            ("ry", [PI/2], "t"),
            ("rx", [PI/2], "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("rx", None, "t"),
            ("ry", [-PI/2], "t"),
            ("sqiswap", [], "*"),
            ("rx", [-PI/2], "c"),
            ("sqiswap", [], "*"),
            ("ry", [PI/2], "t"),
            ("rx", [-PI/2], "t"),
            ("ry", [-PI/2], "t"),
        ]
        # Special case 3: CU needs to be decomposed at a high level
    }
    # "cu": self._cu

    def __init__(self):
        """Add the main initialization table that will drive the transpilation process.

        A value of `None` in the table indicates that parameters from the instruction
        should be used instead.
        """
        super().__init__()

    def transpile_gate(self, instruction: Instruction) -> List[Gate]:
        """Transpile an instruction into a sequence of gates. The result is a list
        since gate ordering matters. The process resembles a dispatch. We use already
        mapped (and swapped) qubits.

        :param instruction: The instruction to transpile.
        :return: A list of gates implementing that instruction.
        """
        gate_maker = self._synthesize(instruction)
        return list(map(lambda g: gate_maker(*g), self._table[instruction.symbol]))

    def transpile_test(self, instruction: Instruction) -> List[Test]:
        pass

    @staticmethod
    def _synthesize(instruction: Instruction) -> Callable[[str, Optional[List[float]]], Gate]:
        """Synthesis method that produces a function which, with the right parameters, yields a gate.

        :param instruction: instruction used to synthesize one gate in the corresponding sequence
        :return: curried function that, upon parameters, completes the gates
        """
        def gate(symbol: str, params: Optional[List[float]] = None, route: str = ".") -> Gate:
            if route in (".", "t"):
                # We add . to distinguish single-qubit gates from two-qubit gates
                tq = instruction.target_qubits
                cq = None
            elif route == "c":
                tq = instruction.control_qubits
                cq = None
            elif route == "*":
                tq = instruction.target_qubits
                cq = instruction.control_qubits
            elif route == "+":
                tq = instruction.control_qubits
                cq = instruction.target_qubits
            else:
                raise ValueError(f"Unsupported routing directive: {route}")

            return Gate(
                symbol=symbol,
                target_qubits=tq,
                control_qubits=cq,
                params=params if params is not None else instruction.parameters,
            )

        return gate