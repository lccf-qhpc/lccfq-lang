"""Qubit movement primitives: swaps and entangling steps."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def swap(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Swap qubits.

    With ``target`` of length 2: a single SWAP between target[0] and target[1].
    With ``pairs`` kwarg: a sequence of independent SWAPs (the ``target`` list
    just declares the qubits that will be touched; every pair must be a subset
    of target).

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :param kwargs:
        pairs: optional list of (a, b) index pairs. When omitted, target must
               have length 2 and a single SWAP is emitted.
    :return: list of SWAP instructions
    """
    pairs = kwargs.get("pairs")

    if pairs is None:
        if len(target) != 2:
            raise ValueError(
                f"swap without 'pairs' requires len(target) == 2, got {len(target)}"
            )
        return [isa.swap(tg_a=target[0], tg_b=target[1])]

    target_set = set(target)
    instructions = []

    for a, b in pairs:
        if a == b:
            raise ValueError(f"swap pair must use distinct qubits, got ({a}, {b})")
        if a not in target_set or b not in target_set:
            raise ValueError(
                f"swap pair ({a}, {b}) not a subset of target {target}"
            )
        instructions.append(isa.swap(tg_a=a, tg_b=b))

    return instructions


def entangle_step(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply an entangling layer across target qubits.

    Topologies:
        "linear"     — CX(target[i], target[i+1]) for i = 0 .. n-2
        "ring"       — linear plus CX(target[-1], target[0])
        "pairs"      — disjoint pairs (target[0],target[1]),
                       (target[2],target[3]), ... ; last qubit is skipped if n odd
        "brickwork"  — even layer then odd layer of disjoint CX
        "all_to_all" — CX(target[i], target[j]) for every i<j

    :param isa: instruction set architecture
    :param target: list of qubit indices (at least 2)
    :param kwargs:
        topology: one of the above (default "linear")
        gate: two-qubit non-parametric ISA gate name (default "cx"); must
              be one of "cx", "cy", "cz", "ch"
    :return: list of entangling instructions
    """
    topology = kwargs.get("topology", "linear")
    gate_name = kwargs.get("gate", "cx")
    n = len(target)

    if n < 2:
        raise ValueError(f"entangle_step requires at least 2 qubits, got {n}")

    if gate_name not in ("cx", "cy", "cz", "ch"):
        raise ValueError(
            f"gate must be one of 'cx', 'cy', 'cz', 'ch', got '{gate_name}'"
        )

    gate_fn = getattr(isa, gate_name)
    instructions = []

    if topology == "linear":
        for i in range(n - 1):
            instructions.append(gate_fn(ct=target[i], tg=target[i + 1]))
    elif topology == "ring":
        for i in range(n - 1):
            instructions.append(gate_fn(ct=target[i], tg=target[i + 1]))
        if n > 2:
            instructions.append(gate_fn(ct=target[-1], tg=target[0]))
    elif topology == "pairs":
        for i in range(0, n - 1, 2):
            instructions.append(gate_fn(ct=target[i], tg=target[i + 1]))
    elif topology == "brickwork":
        for i in range(0, n - 1, 2):
            instructions.append(gate_fn(ct=target[i], tg=target[i + 1]))
        for i in range(1, n - 1, 2):
            instructions.append(gate_fn(ct=target[i], tg=target[i + 1]))
    elif topology == "all_to_all":
        for i in range(n):
            for j in range(i + 1, n):
                instructions.append(gate_fn(ct=target[i], tg=target[j]))
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            f"Expected one of: linear, ring, pairs, brickwork, all_to_all"
        )

    return instructions
