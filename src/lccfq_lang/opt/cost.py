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
import math
import networkx as nx
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING
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
    """Immutable cost record for a quantum program."""
    depth: int
    count_1q: int
    count_2q: int
    count_native_2q: int
    estimated_error: Optional[float] = None

    @classmethod
    def measure(
        cls,
        program: List[Any],
        kind: Literal["arch", "mach"],
        qpu_config: Optional["QPUConfig"] = None,
    ) -> "Cost":
        """Compute the Cost of *program*.

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

        # --- Gate counts ---
        count_1q = 0
        count_2q = 0

        native_set: frozenset = frozenset()
        if kind == "mach" and qpu_config is not None:
            native_set = frozenset(getattr(qpu_config, "native_2q", ()))

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

        # --- Depth ---
        g = circuit_to_dag(program)
        if g.number_of_nodes() > 0:
            depth = nx.dag_longest_path_length(g) + 1
        else:
            depth = 0

        # --- Estimated error ---
        if cal:
            per_gate = cal.get("per_gate_error", {})
            log_sum = math.fsum(
                math.log(1.0 - min(per_gate.get(v.symbol, 0.0), 0.999999))
                for v in qualifying_ops
            )
            estimated_error = math.exp(log_sum)
        else:
            estimated_error = None

        return cls(depth, count_1q, count_2q, count_native_2q, estimated_error)

    def scalarize(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Collapse this Cost to a single scalar score using *weights*.

        Does not mutate *weights*.
        """
        w = weights or DEFAULT_WEIGHTS
        score = (
            w["depth"]           * self.depth
            + w["count_1q"]      * self.count_1q
            + w["count_2q"]      * self.count_2q
            + w["count_native_2q"] * self.count_native_2q
            + (w["error"] * (1.0 - self.estimated_error)
               if self.estimated_error is not None else 0.0)
        )
        return score
