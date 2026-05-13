"""Hamiltonian evolution primitives: continuous time evolution and Trotter decomposition."""

import numpy as np

from typing import Dict, List, Tuple

from ..arch.instruction import Instruction
from ..arch.isa import ISA


PauliTerm = Tuple[float, Dict[int, str]]


def _normalize_paulis(paulis: Dict[int, str], n: int) -> Dict[int, str]:
    """Validate a Pauli dict and drop identity entries.

    Keys are positions in [0, n); values are one of 'I', 'X', 'Y', 'Z'
    (case-insensitive). Identity entries ('I') are dropped silently.
    """
    out = {}
    for pos, p in paulis.items():
        if not isinstance(pos, int) or not 0 <= pos < n:
            raise ValueError(
                f"Pauli position {pos!r} not in [0, {n})"
            )
        if not isinstance(p, str):
            raise TypeError(
                f"Pauli value must be a string, got {type(p).__name__}"
            )
        pu = p.upper()
        if pu not in ("I", "X", "Y", "Z"):
            raise ValueError(
                f"Pauli value must be one of 'I', 'X', 'Y', 'Z'; got '{p}'"
            )
        if pu != "I":
            out[pos] = pu
    return out


def _pauli_string_evolution(
    isa: ISA, target, paulis: Dict[int, str], angle: float
) -> List[Instruction]:
    """Build exp(-i * (angle/2) * P) where P is a tensor product of Paulis.

    Pattern: rotate each qubit into the Z eigenbasis (H for X, Rx(pi/2) for Y),
    fan in via a CNOT ladder so the parity sits on one qubit, apply Rz(angle)
    there, then reverse the CNOT ladder and basis rotations.

    ``paulis`` must already be normalized (no 'I' entries).
    """
    if not paulis:
        return []

    instructions = []
    # Sort by position so the qubit order is deterministic.
    positions = sorted(paulis.keys())
    qubits = [target[p] for p in positions]
    parity_q = qubits[-1]

    # Basis change to Z eigenbasis.
    for pos in positions:
        p = paulis[pos]
        q = target[pos]
        if p == "X":
            instructions.append(isa.h(tg=q))
        elif p == "Y":
            instructions.append(isa.rx(tg=q, params=[np.pi / 2]))

    # CNOT ladder onto parity_q.
    for q in qubits[:-1]:
        instructions.append(isa.cx(ct=q, tg=parity_q))

    # Rz(angle) realizes exp(-i (angle/2) Z) on parity_q, which lifts to
    # exp(-i (angle/2) P) on the full string.
    instructions.append(isa.rz(tg=parity_q, params=[angle]))

    # Reverse CNOT ladder.
    for q in reversed(qubits[:-1]):
        instructions.append(isa.cx(ct=q, tg=parity_q))

    # Reverse basis change.
    for pos in positions:
        p = paulis[pos]
        q = target[pos]
        if p == "X":
            instructions.append(isa.h(tg=q))
        elif p == "Y":
            instructions.append(isa.rx(tg=q, params=[-np.pi / 2]))

    return instructions


def time_evolution(isa: ISA, target, **kwargs) -> List[Instruction]:
    """First-order Trotter step approximation of exp(-i H t).

    The Hamiltonian H is given as a list of (coefficient, paulis) terms:
        H = sum_k coefficient_k * P_k
    where each P_k is a tensor product of Paulis specified by a dict mapping
    positions in target (0 .. n-1) to 'X' / 'Y' / 'Z' (omitted positions and
    'I' values are identity). Each term is evolved sequentially as
    exp(-i c_k P_k t); for non-commuting terms this is an O(t^2)
    approximation, which is exact at t -> 0 and exact for any t when the
    terms commute.

    :param isa: instruction set architecture
    :param target: list of qubit indices the Hamiltonian acts on
    :param kwargs:
        hamiltonian: list of (float, dict[int, str]) terms
        time: float, total evolution time
    :return: list of instructions implementing the first-order step
    """
    hamiltonian = kwargs["hamiltonian"]
    time = float(kwargs["time"])
    n = len(target)

    if n < 1:
        raise ValueError("time_evolution requires at least 1 target qubit")

    instructions = []
    for coef, paulis in hamiltonian:
        c = float(coef)
        active = _normalize_paulis(paulis, n)
        if not active or c == 0.0 or time == 0.0:
            continue
        # Rz(theta) = exp(-i (theta/2) Z), so for exp(-i c P t) we want
        # theta = 2 * c * t on the Z-rotated parity qubit.
        angle = 2.0 * c * time
        instructions.extend(
            _pauli_string_evolution(isa, target, active, angle)
        )

    return instructions


def trotter_steps(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Multi-step Trotter approximation of exp(-i H t).

    Splits the total evolution into ``steps`` substeps of time t/steps and
    applies a first- or second-order Lie-Trotter formula on each substep.
    For non-commuting H, larger ``steps`` reduces the error.

    Order 1 (Lie): per substep, apply each term in list order.
        Error per substep: O((dt)^2). Total error: O(t^2 / steps).
    Order 2 (Strang): per substep, sweep terms forward at dt/2 then
        backward at dt/2. Error per substep: O((dt)^3). Total error:
        O(t^3 / steps^2).

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :param kwargs:
        hamiltonian: list of (coefficient, paulis) terms (see time_evolution)
        time: total evolution time
        steps: number of Trotter substeps (>= 1)
        order: 1 (default) or 2
    :return: list of instructions implementing the Trotterized evolution
    """
    hamiltonian = kwargs["hamiltonian"]
    time = float(kwargs["time"])
    steps = int(kwargs["steps"])
    order = int(kwargs.get("order", 1))
    n = len(target)

    if n < 1:
        raise ValueError("trotter_steps requires at least 1 target qubit")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    if time == 0.0 or not hamiltonian:
        return []

    dt = time / steps
    instructions = []

    if order == 1:
        single = time_evolution(
            isa, target, hamiltonian=hamiltonian, time=dt
        )
        for _ in range(steps):
            instructions.extend(single)
    else:
        # Strang: forward at dt/2, then backward at dt/2.
        half = dt / 2.0
        forward = time_evolution(
            isa, target, hamiltonian=hamiltonian, time=half
        )
        backward = time_evolution(
            isa, target, hamiltonian=list(reversed(hamiltonian)),
            time=half,
        )
        for _ in range(steps):
            instructions.extend(forward)
            instructions.extend(backward)

    return instructions
