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
    """Uniform view over a single operation, regardless of IR level.

    Perf #9: tuple-returning properties (targets, controls, qubits, params)
    are computed lazily on first access and cached on the instance via
    __slots__. Repeat reads cost one attribute fetch.
    """

    __slots__ = ("op", "_kind", "_targets", "_controls", "_qubits", "_params")

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
        # Sentinel: None means "not yet computed" (empty tuple () is a legal
        # value for all four properties, so we cannot use it as a sentinel).
        self._targets = None
        self._controls = None
        self._qubits = None
        self._params = None

    @property
    def symbol(self) -> str:
        return self.op.symbol

    @property
    def targets(self) -> Tuple[int, ...]:
        if self._targets is None:
            self._targets = (
                tuple(self.op.target_qubits or ())
                if self._kind in ("arch", "mach.gate") else ()
            )
        return self._targets

    @property
    def controls(self) -> Tuple[int, ...]:
        if self._controls is None:
            self._controls = (
                tuple(self.op.control_qubits or ())
                if self._kind in ("arch", "mach.gate") else ()
            )
        return self._controls

    @property
    def params(self) -> Tuple[Any, ...]:
        if self._params is None:
            self._params = tuple(self.op.params or ())
        return self._params

    @property
    def qubits(self) -> Tuple[int, ...]:
        if self._qubits is None:
            self._qubits = self.controls + self.targets
        return self._qubits

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

    @property
    def tags(self) -> dict:
        return getattr(self.op, "tags", {})

    @property
    def duration(self):
        return getattr(self.op, "duration", None)

    def __repr__(self) -> str:
        return f"OpView({self._kind}, {self.symbol}, q={self.qubits})"
