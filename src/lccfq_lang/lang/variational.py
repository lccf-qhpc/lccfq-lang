"""Variational ansatz primitives: hardware-efficient ansatz and QAOA step."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA
from .evolution import time_evolution
from .movement import entangle_step


def hw_eff_ansatz(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Hardware-efficient ansatz: alternating layers of single-qubit
    rotations and a fixed entangling pattern.

    Structure: an initial rotation layer, then ``layers`` repetitions of
    (entangling layer + rotation layer). For n qubits, L layers and a
    rotations spec of length R the total parameter count is (L + 1) * n * R.
    Parameters are bound in lexicographic (layer, qubit_position, rotation)
    order.

    :param isa: instruction set architecture
    :param target: list of qubit indices (length n >= 1)
    :param kwargs:
        params: flat list of (L + 1) * n * len(rotations) angles
        layers: number of entangle+rotation blocks after the initial
                rotation layer (>= 0; default 1)
        rotations: string of axes for the single-qubit rotation block,
                   each char in 'x'/'y'/'z' (default "y")
        entangler: topology for entangle_step; default "linear"
        entangle_gate: two-qubit non-parametric gate for entangle_step;
                       default "cx"
    :return: list of instructions implementing the ansatz
    """
    params = list(kwargs["params"])
    layers = int(kwargs.get("layers", 1))
    rotations = kwargs.get("rotations", "y").lower()
    entangler = kwargs.get("entangler", "linear")
    entangle_gate = kwargs.get("entangle_gate", "cx")
    n = len(target)

    if n < 1:
        raise ValueError("hw_eff_ansatz requires at least 1 qubit")
    if layers < 0:
        raise ValueError(f"layers must be >= 0, got {layers}")

    if not rotations:
        raise ValueError("rotations spec must not be empty")
    for c in rotations:
        if c not in ("x", "y", "z"):
            raise ValueError(
                f"rotations must contain only 'x', 'y', 'z'; got '{c}'"
            )

    expected = (layers + 1) * n * len(rotations)
    if len(params) != expected:
        raise ValueError(
            f"params length {len(params)} != expected (L+1) * n * |rotations| "
            f"= ({layers + 1}) * {n} * {len(rotations)} = {expected}"
        )

    instructions = []
    p_iter = iter(params)

    def rotation_layer():
        for q in target:
            for axis in rotations:
                gate_fn = getattr(isa, f"r{axis}")
                instructions.append(
                    gate_fn(tg=q, params=[next(p_iter)])
                )

    def entangle_layer():
        # entangle_step needs n >= 2; for n == 1 there is nothing to entangle.
        if n < 2:
            return
        instructions.extend(
            entangle_step(
                isa, target, topology=entangler, gate=entangle_gate
            )
        )

    # Initial rotation layer, then L (entangler + rotation) blocks.
    rotation_layer()
    for _ in range(layers):
        entangle_layer()
        rotation_layer()

    return instructions


def qaoa_step(isa: ISA, target, **kwargs) -> List[Instruction]:
    """One QAOA layer: exp(-i beta H_M) exp(-i gamma H_C).

    Applies the cost evolution exp(-i gamma * H_cost) followed by the mixer
    evolution exp(-i beta * H_mixer). Each Hamiltonian uses the Pauli-string
    format from time_evolution; if ``mixer`` is omitted it defaults to the
    transverse field H_M = sum_q X_q over every qubit in target.

    Note: when cost terms do not commute, the cost-evolution segment is a
    first-order Trotter step. For typical Ising-like cost Hamiltonians (only
    Z terms) the segment is exact.

    :param isa: instruction set architecture
    :param target: list of qubit indices (n >= 1)
    :param kwargs:
        gamma: float, cost angle
        beta: float, mixer angle
        cost: list of (coef, paulis) terms (see time_evolution)
        mixer: list of (coef, paulis) terms; default = transverse X field
    :return: list of instructions implementing the layer
    """
    gamma = float(kwargs["gamma"])
    beta = float(kwargs["beta"])
    cost = kwargs["cost"]
    n = len(target)

    if n < 1:
        raise ValueError("qaoa_step requires at least 1 qubit")

    mixer = kwargs.get("mixer")
    if mixer is None:
        mixer = [(1.0, {p: "X"}) for p in range(n)]

    instructions = []
    instructions.extend(
        time_evolution(isa, target, hamiltonian=cost, time=gamma)
    )
    instructions.extend(
        time_evolution(isa, target, hamiltonian=mixer, time=beta)
    )
    return instructions
