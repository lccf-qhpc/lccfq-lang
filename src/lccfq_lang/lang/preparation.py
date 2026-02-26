"""
Filename: preparation.py
Author: Santiago Nunez-Corrales
Date: 2026-02-26
Version: 1.0
Description:
    State preparation blocks: basis, uniform superposition, and arbitrary state.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""

import numpy as np

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def prepare_basis(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Prepare a computational basis state in the Z, X, or Y basis.

    Z basis: |0⟩/|1⟩  — X where bit is 1
    X basis: |+⟩/|−⟩  — X where bit is 1, then H on every target
    Y basis: |+i⟩/|−i⟩ — X where bit is 1, then H → S on every target

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :param kwargs:
        bitstring: str of '0'/'1', length must equal len(target)
        basis: "Z" (default), "X", or "Y"
        endianness: "little" (default) or "big"
    :return: list of instructions preparing the basis state
    """
    bitstring = kwargs["bitstring"]
    basis = kwargs.get("basis", "Z").upper()
    endianness = kwargs.get("endianness", "little")
    n = len(target)

    if len(bitstring) != n:
        raise ValueError(
            f"Bitstring length {len(bitstring)} != target count {n}"
        )

    if not all(b in ("0", "1") for b in bitstring):
        raise ValueError(
            f"Bitstring must contain only '0' and '1', got '{bitstring}'"
        )

    if basis not in ("Z", "X", "Y"):
        raise ValueError(
            f"Basis must be 'Z', 'X', or 'Y', got '{basis}'"
        )

    bits = list(bitstring)

    if endianness == "big":
        bits = list(reversed(bits))

    instructions = []

    # Step 1: flip qubits where bit is '1'
    for i, b in enumerate(bits):
        if b == "1":
            instructions.append(isa.x(tg=target[i]))

    # Step 2: rotate into the requested basis
    if basis == "X":
        for i in range(n):
            instructions.append(isa.h(tg=target[i]))
    elif basis == "Y":
        for i in range(n):
            instructions.append(isa.h(tg=target[i]))
            instructions.append(isa.s(tg=target[i]))

    return instructions


def prepare_uniform(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Prepare a uniform superposition over a subset of target qubits.

    Applies H to each qubit in `qubits` (or all of `target` if omitted),
    leaving the rest in |0⟩.

    :param isa: instruction set architecture
    :param target: list of qubit indices in the register
    :param kwargs:
        qubits: list of qubit indices to superpose (must be a subset of
                target); defaults to all of target
    :return: list of H instructions
    """
    qubits = kwargs.get("qubits", target)

    if not set(qubits).issubset(set(target)):
        raise ValueError(
            f"qubits {qubits} is not a subset of target {target}"
        )

    return [isa.h(tg=q) for q in qubits]


def prepare_state(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Prepare an arbitrary quantum state |psi⟩ on target qubits.

    Uses a disentangling decomposition (Möttönen et al.): compute
    uniformly controlled Ry/Rz rotations that reduce |psi⟩ to |0...0⟩,
    then reverse the circuit to obtain the preparation.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :param kwargs:
        state: complex amplitude vector of length 2^len(target)
        endianness: "little" (default) or "big"
    :return: list of instructions preparing the state
    """
    state = np.array(kwargs["state"], dtype=complex)
    endianness = kwargs.get("endianness", "little")
    n = len(target)
    dim = 1 << n

    if state.shape != (dim,):
        raise ValueError(
            f"State vector length {len(state)} != 2^{n} = {dim}"
        )

    norm = np.linalg.norm(state)

    if norm < 1e-15:
        raise ValueError("State vector has zero norm")

    state = state / norm

    if endianness == "big":
        target = list(reversed(target))

    # Phase 1: disentangle from the last qubit to the first.
    # At level k (processed in order k = n-1, n-2, ..., 0) the active
    # entries are 0 .. 2^{k+1}-1 with higher bits already zeroed.
    # Pair entries that differ only in bit k and compute the Ry/Rz
    # angles that zero the bit-k=1 partner.
    omega = state.copy()
    levels = []

    for k in reversed(range(n)):
        half = 1 << k
        ry_angles = []
        rz_angles = []

        for c in range(half):
            i0 = c
            i1 = c + half
            a0, a1 = omega[i0], omega[i1]
            r0, r1 = abs(a0), abs(a1)
            r = np.sqrt(r0 ** 2 + r1 ** 2)

            theta = 2.0 * np.arctan2(r1, r0) if r > 1e-15 else 0.0
            phi = (
                np.angle(a1) - np.angle(a0)
                if r0 > 1e-15 and r1 > 1e-15
                else 0.0
            )

            ry_angles.append(theta)
            rz_angles.append(phi)

            # Update: disentangling maps (a0, a1) → (r·e^{i(α+beta)/2}, 0)
            if r > 1e-15:
                gamma = (np.angle(a0) + np.angle(a1)) / 2.0
                omega[i0] = r * np.exp(1j * gamma)
                omega[i1] = 0.0

        levels.append((k, ry_angles, rz_angles))

    # Phase 2: build the preparation circuit (reverse of disentangling).
    # Disentangling at level k applied  Ry(−theta) ∘ Rz(−phi);
    # preparation applies  Rz(phi) ∘ Ry(theta)  — Ry first in time order.
    instructions = []

    for k, ry_angles, rz_angles in reversed(levels):
        controls = [target[j] for j in range(k)]
        tgt = target[k]

        instructions.extend(
            _ucr(isa, "ry", tgt, controls, ry_angles)
        )

        if any(abs(a) > 1e-15 for a in rz_angles):
            instructions.extend(
                _ucr(isa, "rz", tgt, controls, rz_angles)
            )

    return instructions


def _ucr(isa: ISA, gate_type, target, controls, angles) -> List[Instruction]:
    """Decompose a uniformly controlled rotation into CX + single-qubit gates.

    Uses the recursive multiplexor identity:
        UCR(theta₀..theta_{2^k-1}) = UCR(α) · CX(last_ctrl, tgt) · UCR(beta) · CX
    where alpha_j = (theta_j + theta_{j+half})/2,  beta_j = (theta_j − theta_{j+half})/2.

    :param isa: instruction set architecture
    :param gate_type: "ry" or "rz"
    :param target: target qubit index
    :param controls: list of control qubit indices
    :param angles: rotation angles, one per control basis state
    :return: list of elementary instructions
    """
    # Skip entirely when all angles are negligible
    if all(abs(a) < 1e-15 for a in angles):
        return []

    gate_fn = getattr(isa, gate_type)

    # Base case: no controls → single rotation
    if not controls:
        return [gate_fn(tg=target, params=[angles[0]])]

    half = len(angles) // 2
    alpha = [(angles[j] + angles[j + half]) / 2.0 for j in range(half)]
    beta = [(angles[j] - angles[j + half]) / 2.0 for j in range(half)]

    result = []
    result.extend(_ucr(isa, gate_type, target, controls[:-1], alpha))
    result.append(isa.cx(ct=controls[-1], tg=target))
    result.extend(_ucr(isa, gate_type, target, controls[:-1], beta))
    result.append(isa.cx(ct=controls[-1], tg=target))

    return result
