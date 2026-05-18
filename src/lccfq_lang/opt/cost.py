"""
Filename: cost.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Cost model for quantum programs: gate counts, circuit depth, and
    calibration-driven error estimates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import functools
import math
import networkx as nx
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING
from .op_view import OpView
from .dag import circuit_to_dag

if TYPE_CHECKING:
    from lccfq_lang.sys.base import QPUConfig

DEFAULT_WEIGHTS: Dict[str, float] = {
    "depth": 1.0,
    "count_1q": 0.1,
    "count_2q": 1.0,
    "count_native_2q": 0.5,
    "error": 100.0,
}


@dataclass(frozen=True)
class Cost:
    """Immutable cost record for a quantum program.

    ``depth`` is ``None`` when produced by :meth:`measure_counts` (the cheap
    path that skips DAG construction).  A value of ``0`` means the program
    was measured and found to be empty; ``int >= 1`` is a real circuit depth.
    """
    depth: Optional[int]
    count_1q: int
    count_2q: int
    count_native_2q: int
    estimated_error: Optional[float] = None

    # ------------------------------------------------------------------
    # Private shared helper (Perf #1)
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_counts_and_error(
        program: List[Any],
        kind: Literal["arch", "mach"],
        qpu_config: Optional["QPUConfig"],
    ) -> Tuple[int, int, int, Optional[float]]:
        """Return (count_1q, count_2q, count_native_2q, estimated_error).

        Performs a single linear scan over *program* with OpView — no DAG.
        Assumes *program* is non-empty (callers must guard for empty programs
        before calling this helper).
        """
        cal = getattr(qpu_config, "calibration", None)

        native_set: frozenset = frozenset()
        if kind == "mach" and qpu_config is not None:
            native_set = frozenset(getattr(qpu_config, "native_2q", ()))

        count_1q = 0
        count_2q = 0
        qualifying_ops = []  # for error estimation

        for op in program:
            view = OpView(op)
            if view.is_classical or view.is_measurement:
                continue
            qualifying_ops.append(view)
            nq = len(view.qubits)
            if nq == 1:
                count_1q += 1
            elif view.is_two_qubit:
                count_2q += 1

        # native_2q count
        if kind == "arch":
            count_native_2q = 0
        else:
            if native_set:
                count_native_2q = sum(
                    1 for v in qualifying_ops
                    if v.is_two_qubit and v.symbol in native_set
                )
            else:
                # No native set specified — all 2q ops are "native"
                count_native_2q = count_2q

        # Estimated error
        if cal:
            per_gate = cal.get("per_gate_error", {})
            log_sum = math.fsum(
                math.log(1.0 - min(per_gate.get(v.symbol, 0.0), 0.999999))
                for v in qualifying_ops
            )
            estimated_error: Optional[float] = math.exp(log_sum)
        else:
            estimated_error = None

        return count_1q, count_2q, count_native_2q, estimated_error

    # ------------------------------------------------------------------
    # Public measurement API
    # ------------------------------------------------------------------

    @classmethod
    def measure(
        cls,
        program: List[Any],
        kind: Literal["arch", "mach"],
        qpu_config: Optional["QPUConfig"] = None,
    ) -> "Cost":
        """Compute the full Cost of *program*, including circuit depth.

        Parameters
        ----------
        program:
            List of arch.Instruction or mach.ir.{Gate,Control,Test} objects.
        kind:
            "arch" or "mach" — controls native_2q counting semantics.
        qpu_config:
            Optional QPU configuration; used for native gate set and
            calibration data.
        """
        cal = getattr(qpu_config, "calibration", None)

        # Special-case: empty program
        if not program:
            if cal:
                return cls(0, 0, 0, 0, 1.0)
            return cls(0, 0, 0, 0, None)

        count_1q, count_2q, count_native_2q, estimated_error = (
            cls._measure_counts_and_error(program, kind, qpu_config)
        )

        # --- Depth (requires DAG) ---
        g = circuit_to_dag(program)
        if g.number_of_nodes() > 0:
            depth = nx.dag_longest_path_length(g) + 1
        else:
            depth = 0

        return cls(depth, count_1q, count_2q, count_native_2q, estimated_error)

    @classmethod
    def measure_counts(
        cls,
        program: List[Any],
        kind: Literal["arch", "mach"],
        qpu_config: Optional["QPUConfig"] = None,
    ) -> "Cost":
        """Cheap variant of :meth:`measure`: counts + estimated_error, no DAG, no depth.

        Returns a Cost with ``depth=None``.  Caller is responsible for knowing
        that ``None`` means "not measured" (not "empty program" — that is
        ``depth=0``).  Use :meth:`measure` when depth is required (e.g.
        group-level telemetry).

        This is **Perf #1**: eliminates the NetworkX DAG build for per-pass
        cost telemetry inside the pass manager, where depth is not load-bearing
        for fixpoint termination or per-pass reporting.
        """
        cal = getattr(qpu_config, "calibration", None)

        # Mirror measure() empty-program branch but keep depth=None.
        if not program:
            if cal:
                return cls(None, 0, 0, 0, 1.0)
            return cls(None, 0, 0, 0, None)

        count_1q, count_2q, count_native_2q, estimated_error = (
            cls._measure_counts_and_error(program, kind, qpu_config)
        )
        return cls(None, count_1q, count_2q, count_native_2q, estimated_error)

    # ------------------------------------------------------------------
    # Scalarization
    # ------------------------------------------------------------------

    def scalarize(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Collapse this Cost to a single scalar score using *weights*.

        When ``depth is None`` (cheap-path measurement), the depth term is
        omitted from the sum — symmetric with the ``estimated_error is None``
        handling.  No callers of ``scalarize`` need to change.

        Does not mutate *weights*.

        Perf #10: the no-weights (DEFAULT_WEIGHTS) path is the only
        argument shape used by PassManager fixpoint-termination compares;
        cache it via module-level lru_cache since Cost is a frozen dataclass
        (hashable). Custom-weights path computes fresh.
        """
        if weights is None:
            return _scalarize_default(self)
        w = weights
        score = (
            (w["depth"] * self.depth if self.depth is not None else 0.0)
            + w["count_1q"]      * self.count_1q
            + w["count_2q"]      * self.count_2q
            + w["count_native_2q"] * self.count_native_2q
            + (w["error"] * (1.0 - self.estimated_error)
               if self.estimated_error is not None else 0.0)
        )
        return score


@functools.lru_cache(maxsize=1024)
def _scalarize_default(cost: "Cost") -> float:
    """Cached default-weights scalarization. Module-level so it can be
    decorated; Cost is frozen (hashable) so the cache key is well-defined."""
    w = DEFAULT_WEIGHTS
    return (
        (w["depth"] * cost.depth if cost.depth is not None else 0.0)
        + w["count_1q"]      * cost.count_1q
        + w["count_2q"]      * cost.count_2q
        + w["count_native_2q"] * cost.count_native_2q
        + (w["error"] * (1.0 - cost.estimated_error)
           if cost.estimated_error is not None else 0.0)
    )
