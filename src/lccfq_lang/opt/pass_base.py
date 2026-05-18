"""
Filename: pass_base.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Abstract base classes and data structures for the optimization pass
    infrastructure in lccfq-lang.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from lccfq_lang.sys.base import QPUConfig
    from lccfq_lang.arch.isa import ISA
    from lccfq_lang.arch.mapping import QPUMapping
    from .cost import Cost

Program = List[Any]
AppliesTo = Literal["arch", "mach"]
PassResult = Tuple[Program, bool]


@dataclass
class PassContext:
    """Read-mostly context object passed to every Pass.run() call."""
    qpu_config: Optional["QPUConfig"] = None
    isa: Optional["ISA"] = None
    mapping: Optional["QPUMapping"] = None
    topology: Optional[Any] = None
    cost_before: Optional["Cost"] = None
    scratchpad: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PassRecord:
    """Per-pass execution telemetry returned by PassManager.run().

    .. note:: **Perf #1 depth semantics**

        ``cost_before.depth`` and ``cost_after.depth`` are ``None`` for:

        - every record produced inside a fixpoint group's inner pass-sweep, and
        - every non-boundary record in a multi-pass linear group (i.e. the
          post-cost of a non-last pass and the pre-cost of a non-first pass).

        ``None`` means "not measured via the DAG path" — it does **not** mean
        the program is empty (that would be ``depth=0``).

        For group-level depth telemetry with real integer depth values, consult
        ``groups[i].cost_before / cost_after`` in ``Circuit.opt_report``, which
        are populated from full ``Cost.measure`` calls at group boundaries.
    """
    pass_name: str
    group_name: str
    iteration: int
    cost_before: "Cost"
    cost_after: "Cost"
    delta_seconds: float
    changed: bool  # Perf #4: True iff this pass reported a transformation.


class Pass(ABC):
    """Abstract base for a single optimization pass.

    A Pass is a pure function: program -> program. Implementations MUST NOT
    mutate the input list or its elements; they MUST return a new list.
    """
    name: str
    applies_to: AppliesTo

    @abstractmethod
    def run(self, program: Program, ctx: PassContext) -> PassResult:
        """Transform `program` and return ``(new_program, changed)``.

        *changed* MUST be ``True`` if the returned ``new_program`` differs
        in any observable way from ``program``. It MAY be ``False`` only
        when the pass guarantees the returned program is semantically AND
        structurally identical to the input (same op count, same op order,
        same op fields). When in doubt, return ``True``.

        Lying (returning ``changed=False`` while having modified the
        program) corrupts post-cost telemetry and is detected by the
        debug-mode audit (``PassManager.debug_assert_changed=True``).
        """
        ...
