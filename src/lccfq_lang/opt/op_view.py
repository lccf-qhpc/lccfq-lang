"""
Filename: op_view.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Uniform view over heterogeneous operation types (arch Instruction and
    mach IR nodes) used by the optimization infrastructure.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from typing import Any, Tuple
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.mach.ir import Gate, Control, Test


class OpView:
    """Uniform view over a single operation, regardless of IR level."""

    __slots__ = ("op", "_kind")

    def __init__(self, op: Any) -> None:
        self.op = op
        if isinstance(op, Instruction):
            self._kind = "arch"
        elif isinstance(op, Gate):
            self._kind = "mach.gate"
        elif isinstance(op, Control):
            self._kind = "mach.control"
        elif isinstance(op, Test):
            self._kind = "mach.test"
        else:
            raise TypeError(f"OpView: unsupported op type {type(op).__name__}")

    @property
    def symbol(self) -> str:
        return self.op.symbol

    @property
    def targets(self) -> Tuple[int, ...]:
        if self._kind in ("arch", "mach.gate"):
            return tuple(self.op.target_qubits or ())
        return ()

    @property
    def controls(self) -> Tuple[int, ...]:
        if self._kind in ("arch", "mach.gate"):
            return tuple(self.op.control_qubits or ())
        return ()

    @property
    def params(self) -> Tuple[Any, ...]:
        return tuple(self.op.params or ())

    @property
    def qubits(self) -> Tuple[int, ...]:
        return tuple(self.controls) + tuple(self.targets)

    @property
    def is_two_qubit(self) -> bool:
        return len(self.qubits) >= 2

    @property
    def is_measurement(self) -> bool:
        return self.symbol == "measure"

    @property
    def is_classical(self) -> bool:
        return self._kind in {"mach.control", "mach.test"}

    @property
    def kind(self) -> str:
        return self._kind

    def __repr__(self) -> str:
        return f"OpView({self._kind}, {self.symbol}, q={self.qubits})"
