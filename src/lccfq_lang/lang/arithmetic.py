"""Quantum arithmetic primitives: addition, modular multiplication, comparison."""

import numpy as np

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA
from .transforms import qft, iqft


def add(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Reversible in-place addition modulo 2^n using the Draper QFT adder.

    Two modes:
        classical+quantum: provide ``value`` (int). The result is
            |b> -> |(b + value) mod 2^n>.
        quantum+quantum:   provide ``register`` (list of n qubit indices).
            The result is |a>|b> -> |a>|(a + b) mod 2^n>; the addend
            register is left unchanged.

    The construction wraps a no-swap QFT around phase rotations:
    qubit ``target[k]`` (little-endian: target[0] is LSB) receives phase
    angle 2π · v / 2^(n-k) where v is either the classical value or
    (in the quantum case) 2^j conditioned on register[j] being 1.

    :param isa: instruction set architecture
    :param target: list of qubit indices that will hold the sum (length n >= 1)
    :param kwargs:
        value: classical integer to add (mutually exclusive with ``register``)
        register: quantum addend, list of n qubit indices (mutually exclusive
                  with ``value``); must be disjoint from target
    :return: list of instructions implementing the addition
    """
    value = kwargs.get("value", None)
    register = kwargs.get("register", None)
    n = len(target)

    if n < 1:
        raise ValueError("add requires at least 1 target qubit")

    if (value is None) == (register is None):
        raise ValueError(
            "add requires exactly one of 'value' or 'register'"
        )

    if register is not None:
        if len(register) != n:
            raise ValueError(
                f"register length {len(register)} != target length {n}"
            )
        if set(register) & set(target):
            raise ValueError(
                f"register and target must be disjoint; overlap: "
                f"{sorted(set(register) & set(target))}"
            )

    instructions = []
    instructions.extend(qft(isa, target, do_swaps=True))

    if value is not None:
        v = int(value) % (1 << n)
        for k in range(n):
            shift = n - k
            angle = 2.0 * np.pi * v / (1 << shift)
            instructions.append(isa.p(tg=target[k], params=[angle]))
    else:
        for k in range(n):
            for j in range(n):
                shift = n - k - j
                if shift <= 0:
                    continue
                angle = 2.0 * np.pi / (1 << shift)
                instructions.append(
                    isa.cp(ct=register[j], tg=target[k], params=[angle])
                )

    instructions.extend(iqft(isa, target, do_swaps=True))

    return instructions


def mult_mod(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Out-of-place modular multiplication by a classical constant.

        |x>_register |0>_target  ->  |x>_register |(value * x) mod modulus>_target

    Only ``modulus == 2**n`` (where n is len(target)) is supported in this
    revision: the multiplication is performed by Draper-style controlled
    phase rotations and the modular reduction is the natural truncation to
    n bits. A non-2**n modulus requires the Vedral-Beckman-Barenco modular
    adder ladder; raise NotImplementedError until that lands.

    :param isa: instruction set architecture
    :param target: list of qubit indices (length n >= 1) holding the result;
                   must start in |0>
    :param kwargs:
        value: classical multiplier a (any integer; taken mod modulus)
        register: list of n quantum qubits holding the multiplicand x
                  (must be disjoint from target)
        modulus: positive integer; only 2**n is supported here, defaults to it
    :return: list of instructions implementing the multiplication
    """
    value = kwargs["value"]
    register = kwargs["register"]
    n = len(target)
    modulus = kwargs.get("modulus", 1 << n)

    if n < 1:
        raise ValueError("mult_mod requires at least 1 target qubit")
    if len(register) != n:
        raise ValueError(
            f"register length {len(register)} != target length {n}"
        )
    if set(register) & set(target):
        raise ValueError(
            f"register and target must be disjoint; overlap: "
            f"{sorted(set(register) & set(target))}"
        )
    if modulus != (1 << n):
        raise NotImplementedError(
            f"mult_mod currently only supports modulus = 2**n ({1 << n}); "
            f"got {modulus}"
        )

    a = int(value) % modulus

    instructions = []
    instructions.extend(qft(isa, target, do_swaps=True))

    # For each pair (register[j], target[k]) the phase contribution on
    # target[k] when register[j] = 1 is 2*pi * a * 2**j / 2**(n-k); skip
    # angles that are integer multiples of 2*pi (shift <= 0).
    for k in range(n):
        for j in range(n):
            shift = n - k - j
            if shift <= 0:
                continue
            angle = 2.0 * np.pi * a / (1 << shift)
            instructions.append(
                isa.cp(ct=register[j], tg=target[k], params=[angle])
            )

    instructions.extend(iqft(isa, target, do_swaps=True))

    return instructions


def compare(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Compare a quantum register against a classical integer.

    XORs the comparison outcome into a single ``result`` qubit:
        result <- result XOR f(target, value)

    Operations:
        "eq" (default): result XOR= (target == value)
        "ne":          result XOR= (target != value)
        "lt":          result XOR= (target < value)
        "ge":          result XOR= (target >= value)

    The ``lt`` and ``ge`` modes are implemented via Draper subtraction on an
    extended register; they require a fresh ``sign_ancilla`` qubit that is
    consumed and restored to |0> by the circuit. ``eq`` and ``ne`` work
    in-place on ``target`` and need no ancilla.

    :param isa: instruction set architecture
    :param target: list of qubit indices holding the operand (length n >= 1,
                   little-endian: target[0] is LSB)
    :param kwargs:
        op: one of "eq", "ne", "lt", "ge" (default "eq")
        value: classical integer to compare against (interpreted mod 2**n
               for eq/ne; clamped semantically for lt/ge)
        result: qubit index that receives the XOR'd outcome; must be disjoint
                from target and sign_ancilla
        sign_ancilla: required for "lt" and "ge"; must start in |0> and is
                      restored to |0> by the circuit; must be disjoint from
                      target and result
    :return: list of instructions implementing the comparison
    """
    op = kwargs.get("op", "eq")
    value = kwargs["value"]
    result = kwargs["result"]
    n = len(target)

    if n < 1:
        raise ValueError("compare requires at least 1 target qubit")
    if op not in ("eq", "ne", "lt", "ge"):
        raise ValueError(
            f"op must be one of 'eq', 'ne', 'lt', 'ge'; got '{op}'"
        )
    if result in target:
        raise ValueError(
            f"result {result} must not appear in target {target}"
        )

    if op in ("eq", "ne"):
        return _compare_eq(isa, target, int(value), result, negate=(op == "ne"))

    sign_ancilla = kwargs.get("sign_ancilla")
    if sign_ancilla is None:
        raise ValueError(
            f"op '{op}' requires a 'sign_ancilla' keyword argument"
        )
    if sign_ancilla in target or sign_ancilla == result:
        raise ValueError(
            f"sign_ancilla {sign_ancilla} must be disjoint from target and result"
        )

    return _compare_lt(
        isa, target, int(value), result, sign_ancilla, negate=(op == "ge")
    )


def _compare_eq(
    isa: ISA, target, value: int, result: int, negate: bool
) -> List[Instruction]:
    n = len(target)
    k = value % (1 << n)

    instructions = []

    # Flip target bits where bit of k is 0 so that target equals all-ones iff
    # the original value equals k.
    flips = [target[p] for p in range(n) if ((k >> p) & 1) == 0]
    for q in flips:
        instructions.append(isa.x(tg=q))

    instructions.append(Instruction(
        symbol="x",
        modifies_state=False,
        is_controlled=True,
        target_qubits=[result],
        control_qubits=list(target),
        params=None,
        shots=None,
    ))

    for q in flips:
        instructions.append(isa.x(tg=q))

    if negate:
        instructions.append(isa.x(tg=result))

    return instructions


def _compare_lt(
    isa: ISA, target, value: int, result: int, sign_ancilla: int,
    negate: bool,
) -> List[Instruction]:
    n = len(target)
    dim = 1 << n

    # Short-circuit out-of-range comparisons.
    if value <= 0:
        # a < k always false → result XOR= 0; "ge" flips it.
        if negate:
            return [isa.x(tg=result)]
        return []
    if value >= dim:
        # a < k always true → result XOR= 1; "ge" flips it back to 0.
        if negate:
            return []
        return [isa.x(tg=result)]

    extended = list(target) + [sign_ancilla]
    addend = (1 << (n + 1)) - value

    instructions = []
    # Subtract value: extended <- (extended + 2^(n+1) - value) mod 2^(n+1)
    instructions.extend(add(isa, extended, value=addend))
    # sign_ancilla now holds (a < value)
    instructions.append(isa.cx(ct=sign_ancilla, tg=result))
    # Restore extended: add value back so sign_ancilla returns to |0>.
    instructions.extend(add(isa, extended, value=value))

    if negate:
        instructions.append(isa.x(tg=result))

    return instructions
