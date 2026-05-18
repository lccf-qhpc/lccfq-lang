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
from typing import Any, Dict, List, Literal, Optional, Tuple
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
        # Single-slot cost cache (Perf #2). Reset on every .run() entry.
        # Key: (id(program), kind, id(qpu_config)); Value: Cost instance.
        # Not thread-safe; one .run() at a time per instance.
        self._last_cost_key: Optional[Tuple[int, str, int]] = None
        self._last_cost_value: Optional[Cost] = None
        self._cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}

    def run(
        self,
        program: List[Any],
        ctx: Optional[PassContext] = None,
    ) -> Tuple[List[Any], List[PassRecord], Dict[str, Tuple[Cost, Cost]]]:
        """Apply all groups to *program* and return the transformed program,
        telemetry records, and group-level boundary Costs for fixpoint groups.

        Returns
        -------
        (program, records, groups_meta)
            program:
                The fully-transformed program list.
            records:
                One :class:`PassRecord` per pass execution.
            groups_meta:
                Mapping from group name to ``(group_cost_before, group_cost_after)``
                for **fixpoint** groups only.  Both Costs in the tuple are produced
                by full :meth:`Cost.measure` calls (with real depth) and are suitable
                for group-level report telemetry.  Linear groups are absent from this
                dict; their group-boundary Costs are derived from the first/last
                records (which use full :meth:`Cost.measure` under Perf #1 decision
                C.2).

        The original *program* list is never mutated; element identity may be
        shared between the input and output.

        .. note::
            **Breaking change (Perf #1):** this method previously returned a
            2-tuple ``(program, records)``.  It now returns a 3-tuple.  The
            only in-tree caller (``arch/context.py``) has been updated.
        """
        # Perf #2: reset per-run cost cache + stats.
        self._last_cost_key = None
        self._last_cost_value = None
        self._cache_stats = {"hits": 0, "misses": 0}

        if ctx is None:
            ctx = PassContext()
        ctx.scratchpad = {}

        current = list(program)
        records: List[PassRecord] = []
        groups_meta: Dict[str, Tuple[Cost, Cost]] = {}

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
                current, gmeta = self._run_fixpoint(group, current, ctx, kind, records)
                groups_meta[group.name] = gmeta

        return (current, records, groups_meta)

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
        """Run all passes in *group* once, in order.

        **Perf #1 (decision C.2):** the first pass's ``cost_before`` and the
        last pass's ``cost_after`` use full :meth:`Cost.measure` (DAG built),
        so that :func:`_build_opt_report` can derive group-level depth from
        ``rs[0].cost_before`` / ``rs[-1].cost_after`` as today.  All other
        per-pass costs use the cheaper :meth:`Cost.measure_counts` (no DAG).
        """
        current = program
        n_passes = len(group.passes)
        for i, p in enumerate(group.passes):
            is_first = (i == 0)
            is_last = (i == n_passes - 1)
            if is_first:
                cost_before = self._measure(current, kind, ctx.qpu_config)
            else:
                cost_before = Cost.measure_counts(current, kind, ctx.qpu_config)
            t0 = time.perf_counter()
            if current:
                new_program = p.run(current, ctx)
            else:
                new_program = current
            delta = time.perf_counter() - t0
            if is_last:
                cost_after = self._measure(new_program, kind, ctx.qpu_config)
            else:
                cost_after = Cost.measure_counts(new_program, kind, ctx.qpu_config)
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
    ) -> Tuple[List[Any], Tuple[Cost, Cost]]:
        """Run passes in *group* in a fixpoint loop until convergence or max_iters.

        Returns
        -------
        (current, (initial_group_cost_before, final_group_cost_after))
            The group-boundary Costs are both produced by full
            :meth:`Cost.measure` calls (with real depth) for use in
            ``groups_meta`` and ultimately :func:`_build_opt_report`.

        **Perf #1:** the outer ``group_cost_before`` / ``group_cost_after``
        remain full :meth:`Cost.measure` calls (fixpoint termination semantics
        preserved; group-level report depth is accurate).  The inner per-pass
        ``cost_before`` / ``cost_after`` use :meth:`Cost.measure_counts` (no
        DAG), eliminating the bulk of DAG builds per compile.
        """
        current = program
        # Full Cost for group boundary — both report accuracy and termination.
        initial_group_cost_before = self._measure(current, kind, ctx.qpu_config)
        group_cost_before = initial_group_cost_before
        # Initialise so the return value is always defined (handles max_iters=0
        # corner case, though PassGroup validation requires max_iters >= 1).
        final_group_cost_after = group_cost_before

        for it in range(group.max_iters):
            # Inner sweep: use cheap measure_counts for per-pass records.
            for p in group.passes:
                cost_before = Cost.measure_counts(current, kind, ctx.qpu_config)
                t0 = time.perf_counter()
                if current:
                    new_program = p.run(current, ctx)
                else:
                    new_program = current
                delta = time.perf_counter() - t0
                cost_after = Cost.measure_counts(new_program, kind, ctx.qpu_config)
                records.append(PassRecord(
                    pass_name=p.name,
                    group_name=group.name,
                    iteration=it,
                    cost_before=cost_before,
                    cost_after=cost_after,
                    delta_seconds=delta,
                ))
                current = new_program

            # Full Cost for termination and group-level report.
            group_cost_after = self._measure(current, kind, ctx.qpu_config)
            final_group_cost_after = group_cost_after
            improvement = (
                group_cost_before.scalarize() - group_cost_after.scalarize()
            )
            if improvement <= group.tol:
                break
            group_cost_before = group_cost_after

        return current, (initial_group_cost_before, final_group_cost_after)

    def _measure(
        self,
        program: List[Any],
        kind: str,
        qpu_config: Optional[Any],
    ) -> Cost:
        """Cached full Cost.measure (Perf #2).

        Single-slot cache keyed by ``(id(program), kind, id(qpu_config))``.
        Cache lifetime: one call to :meth:`PassManager.run`.

        The cache is safe from false hits because the ``current`` local variable
        in :meth:`run` (and the inner ``current`` in :meth:`_run_linear` /
        :meth:`_run_fixpoint`) holds a strong reference to the live program list
        for the full duration of ``.run()``.  The program object whose ``id``
        is stored in ``_last_cost_key`` is therefore guaranteed to be live at
        the point of the next call — its ``id`` cannot have been reused by GC.

        Not thread-safe; one ``.run()`` at a time per instance.

        Stats are accumulated in :attr:`_cache_stats` (``"hits"`` / ``"misses"``)
        for verification testing.
        """
        key = (id(program), kind, id(qpu_config))
        if self._last_cost_key == key:
            self._cache_stats["hits"] += 1
            return self._last_cost_value
        self._cache_stats["misses"] += 1
        cost = Cost.measure(program, kind, qpu_config)
        self._last_cost_key = key
        self._last_cost_value = cost
        return cost
