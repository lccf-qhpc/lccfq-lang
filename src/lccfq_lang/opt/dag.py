"""
Filename: dag.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Build a directed acyclic graph (DAG) view of a program, and reconstruct
    the program from the DAG in topological order.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import heapq
import networkx as nx
from typing import Any, Dict, List
from .op_view import OpView


def circuit_to_dag(program: List[Any]) -> nx.DiGraph:
    """Build a qubit-flow DAG from a list of operations.

    Each node carries the original op object and its index in *program*.
    Edges are labelled with the qubit(s) that create the data dependency.

    Returns an empty DiGraph for an empty program.
    """
    g = nx.DiGraph()

    for i, op in enumerate(program):
        g.add_node(i, op=op, idx=i)

    last_writer: Dict[int, int] = {}

    for i, op in enumerate(program):
        qs = OpView(op).qubits
        for q in qs:
            if q in last_writer:
                j = last_writer[q]
                if g.has_edge(j, i):
                    # Append qubit to existing edge's list (will be normalised below)
                    g[j][i]["qubits"] = list(g[j][i]["qubits"]) + [q]
                else:
                    g.add_edge(j, i, qubits=[q])
            last_writer[q] = i

    # Normalise every edge: sorted, deduplicated tuple
    for u, v in g.edges():
        g[u][v]["qubits"] = tuple(sorted(set(g[u][v]["qubits"])))

    return g


def dag_to_program(g: nx.DiGraph) -> List[Any]:
    """Reconstruct a program from a DAG using Kahn-style topological sort.

    Ties are broken by the original index stored in each node so the output
    order is deterministic and matches the original program when no edges were
    removed.

    Raises ValueError if the graph contains a cycle.
    """
    indeg = {n: g.in_degree(n) for n in g.nodes}
    ready = [(g.nodes[n]["idx"], n) for n in g.nodes if indeg[n] == 0]
    heapq.heapify(ready)
    out = []

    while ready:
        _, n = heapq.heappop(ready)
        out.append(g.nodes[n]["op"])
        for _, m in g.out_edges(n):
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(ready, (g.nodes[m]["idx"], m))

    if len(out) != g.number_of_nodes():
        raise ValueError("dag_to_program: graph contains a cycle")
    return out
