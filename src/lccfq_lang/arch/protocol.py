"""
Filename: protocol.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Structural protocol defining what the arch layer requires from a backend.
    Any object providing these attributes/methods satisfies the contract —
    no explicit inheritance needed.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from typing import Protocol, Dict, List, runtime_checkable

from .isa import ISA
from .instruction import Instruction


class TranspilerLike(Protocol):
    """Minimal transpiler interface needed by the compilation pipeline."""

    def transpile_gate(self, instruction: Instruction) -> List:
        ...


@runtime_checkable
class Backend(Protocol):
    """Structural protocol for a QPU backend as seen by the arch layer.

    Any object exposing these attributes satisfies the contract.
    The concrete ``QPU`` class in ``backend.py`` matches this protocol
    without needing to inherit from it.
    """

    last_pass: str
    isa: ISA
    transpiler: TranspilerLike

    def exec_circuit(self, circuit: List, shots: int) -> Dict[str, float]:
        ...

    def exec_single(self, instruction: Instruction, shots: int):
        ...
