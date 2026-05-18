"""
Filename: _survivor_chain.py
Author: Santiago Nunez-Corrales
Date: 2026-05-18
Version: 1.0
Description:
    Shared helper for peephole passes: per-qubit doubly-linked-list of
    surviving operations.  Replaces O(n) backward-scan _previous_op_on
    with O(1) pointer lookup.

    Invariants:
    - `survivors` is append-only; cancellation sets node.alive = False.
    - Each node participates in exactly one per-qubit chain per qubit it touches.
    - Replace-in-place must preserve the qubit footprint of the replaced op;
      chains are NOT respliced on replace-in-place.  A dev-mode assert guards
      this invariant (see callers).
    - Empty-qubit ops (OpView.qubits == ()) are appended to `survivors` for
      emit-ordering but are not linked into any per-qubit chain.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from lccfq_lang.opt.op_view import OpView


class _SurvivorNode:
    """One node in the append-only survivors list, with per-qubit DLL pointers.

    Invariant: if node.alive is False, prev_per_qubit[q] and next_per_qubit[q]
    are cleared to None for all q (done by cancel_node to fail loudly on stale
    reads).
    """
    __slots__ = ("op", "idx", "alive", "prev_per_qubit", "next_per_qubit")

    def __init__(self, op, idx: int) -> None:
        self.op = op
        self.idx: int = idx
        self.alive: bool = True
        self.prev_per_qubit: dict[int, "_SurvivorNode | None"] = {}
        self.next_per_qubit: dict[int, "_SurvivorNode | None"] = {}


def append_node(
    survivors: list,
    last_node_on_q: dict,
    op,
) -> _SurvivorNode:
    """Create a node for *op*, append it to *survivors*, and thread it into
    the per-qubit chains for every qubit it touches.

    Returns the new node so callers can hold a reference if needed.
    """
    node = _SurvivorNode(op, idx=len(survivors))
    survivors.append(node)
    for q in OpView(op).qubits:
        prev = last_node_on_q.get(q)
        node.prev_per_qubit[q] = prev
        node.next_per_qubit[q] = None
        if prev is not None:
            prev.next_per_qubit[q] = node
        last_node_on_q[q] = node
    return node


def cancel_node(victim: _SurvivorNode, last_node_on_q: dict) -> None:
    """Mark *victim* as cancelled and splice it out of every per-qubit chain
    it belongs to.

    After this call victim.alive is False and all per-qubit pointers on
    *victim* are set to None (defensive: stale reads will return None rather
    than silently following dangling pointers).
    """
    victim.alive = False
    for q in list(victim.prev_per_qubit.keys()):
        p = victim.prev_per_qubit.get(q)
        n = victim.next_per_qubit.get(q)
        if p is not None:
            p.next_per_qubit[q] = n
        if n is not None:
            n.prev_per_qubit[q] = p
        if last_node_on_q.get(q) is victim:
            if p is None:
                last_node_on_q.pop(q, None)
            else:
                last_node_on_q[q] = p
        # Defensive clear.
        victim.prev_per_qubit[q] = None
        victim.next_per_qubit[q] = None
