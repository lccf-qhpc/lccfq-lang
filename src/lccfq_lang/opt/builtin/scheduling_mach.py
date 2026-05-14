"""
Filename: scheduling_mach.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Scheduling/measurement passes for the mach-level IR.

    DeferMeasurement  — pushes measure ops to the end in stable order
    ParallelizeLayers — ASAP layer tagging via circuit-to-DAG analysis

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import networkx as nx
from lccfq_lang.arch.isa import ISA
from lccfq_lang.mach.ir import Gate
from lccfq_lang.opt.pass_base import Pass, PassContext
from lccfq_lang.opt.dag import circuit_to_dag
from ._native import NATIVE_MEASURE


class DeferMeasurement(Pass):
    """Pushes all measure ops to the end of the program in stable
    relative order.

    v1 simplification: classical-conditional ops (Control / Test) are
    NOT split out — they are emitted in their original positions, and
    measure ops slot in AFTER all non-measure ops. This is safe in v1
    because the language has no classical-conditional read of measure
    results between the measure and a subsequent unitary.
    """
    name = "defer_measurement"
    applies_to = "mach"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program, ctx):
        non_measures: list = []
        measures: list = []
        for op in program:
            if isinstance(op, Gate) and op.symbol in NATIVE_MEASURE:
                measures.append(op)
            else:
                non_measures.append(op)
        return non_measures + measures


class ParallelizeLayers(Pass):
    """Analysis-only: tags each Gate with its ASAP layer index in
    Gate.tags["layer"]. Does NOT reorder.

    EXCEPTION TO PURITY: writes into Gate.tags in place. The list
    itself is shallow-copied so callers see a new list identity, but
    Gate identity is preserved. Classical ops (Control, Test) are not
    tagged.
    """
    name = "parallelize_layers"
    applies_to = "mach"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program, ctx):
        if not program:
            return list(program)
        g = circuit_to_dag(program)
        # ASAP layer = longest path length from any source to this node.
        # Equivalent to nx.dag_longest_path_length restricted to the
        # subgraph ending at each node. Compute by topo order.
        layer: dict[int, int] = {}
        for n in nx.topological_sort(g):
            preds = list(g.predecessors(n))
            layer[n] = 0 if not preds else 1 + max(layer[p] for p in preds)
        for n in g.nodes:
            op = g.nodes[n]["op"]
            if isinstance(op, Gate):
                op.tags["layer"] = layer[n]
        return list(program)
