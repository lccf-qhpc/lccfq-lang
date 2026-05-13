"""
Filename: manager.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    PassGroup and PassManager — the driver that sequences optimization passes
    over a program.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.mach.ir import Command
from .pass_base import Pass, PassContext, PassRecord
from .cost import Cost


@dataclass
class PassGroup:
    """A named, ordered collection of passes with a shared execution mode."""
    name: str
    mode: Literal["linear", "fixpoint"]
    passes: List[Pass]
    max_iters: int = 5
    tol: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PassGroup: name must be non-empty")
        if self.mode not in ("linear", "fixpoint"):
            raise ValueError(f"PassGroup {self.name}: invalid mode {self.mode!r}")
        if not self.passes:
            raise ValueError(f"PassGroup {self.name}: passes list must be non-empty")
        applies_to_values = {p.applies_to for p in self.passes}
        if len(applies_to_values) > 1:
            raise ValueError(
                f"PassGroup {self.name}: mixed applies_to values {applies_to_values}"
            )
        if self.max_iters < 1:
            raise ValueError(f"PassGroup {self.name}: max_iters must be >= 1")
        if self.tol < 0.0:
            raise ValueError(f"PassGroup {self.name}: tol must be >= 0.0")


class PassManager:
    """Runs an ordered sequence of PassGroups over a program."""

    def __init__(self, groups: List[PassGroup]) -> None:
        self.groups = groups

    def run(
        self,
        program: List[Any],
        ctx: Optional[PassContext] = None,
    ) -> Tuple[List[Any], List[PassRecord]]:
        """Apply all groups to *program* and return the transformed program and telemetry.

        The original *program* list is never mutated; element identity may be
        shared between the input and output.
        """
        if ctx is None:
            ctx = PassContext()
        ctx.scratchpad = {}

        current = list(program)
        records: List[PassRecord] = []

        for group in self.groups:
            kind = "arch" if group.passes[0].applies_to == "arch" else "mach"
            # Type-check non-empty programs
            if current:
                first = current[0]
                if kind == "arch" and not isinstance(first, Instruction):
                    raise TypeError(
                        f"PassGroup {group.name} expects arch program, "
                        f"got {type(first).__name__}"
                    )
                if kind == "mach" and not isinstance(first, Command):
                    raise TypeError(
                        f"PassGroup {group.name} expects mach program, "
                        f"got {type(first).__name__}"
                    )

            if group.mode == "linear":
                current = self._run_linear(group, current, ctx, kind, records)
            else:
                current = self._run_fixpoint(group, current, ctx, kind, records)

        return (current, records)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_linear(
        self,
        group: PassGroup,
        program: List[Any],
        ctx: PassContext,
        kind: str,
        records: List[PassRecord],
    ) -> List[Any]:
        current = program
        for p in group.passes:
            cost_before = Cost.measure(current, kind, ctx.qpu_config)
            t0 = time.perf_counter()
            if current:
                new_program = p.run(current, ctx)
            else:
                new_program = current
            delta = time.perf_counter() - t0
            cost_after = Cost.measure(new_program, kind, ctx.qpu_config)
            records.append(PassRecord(
                pass_name=p.name,
                group_name=group.name,
                iteration=0,
                cost_before=cost_before,
                cost_after=cost_after,
                delta_seconds=delta,
            ))
            current = new_program
        return current

    def _run_fixpoint(
        self,
        group: PassGroup,
        program: List[Any],
        ctx: PassContext,
        kind: str,
        records: List[PassRecord],
    ) -> List[Any]:
        current = program
        group_cost_before = Cost.measure(current, kind, ctx.qpu_config)

        for it in range(group.max_iters):
            # Inner sweep over passes
            for p in group.passes:
                cost_before = Cost.measure(current, kind, ctx.qpu_config)
                t0 = time.perf_counter()
                if current:
                    new_program = p.run(current, ctx)
                else:
                    new_program = current
                delta = time.perf_counter() - t0
                cost_after = Cost.measure(new_program, kind, ctx.qpu_config)
                records.append(PassRecord(
                    pass_name=p.name,
                    group_name=group.name,
                    iteration=it,
                    cost_before=cost_before,
                    cost_after=cost_after,
                    delta_seconds=delta,
                ))
                current = new_program

            group_cost_after = Cost.measure(current, kind, ctx.qpu_config)
            improvement = (
                group_cost_before.scalarize() - group_cost_after.scalarize()
            )
            if improvement <= group.tol:
                break
            group_cost_before = group_cost_after

        return current
