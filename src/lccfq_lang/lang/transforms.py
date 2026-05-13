"""Unitary transforms: quantum Fourier transform and its inverse."""

import numpy as np

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def qft(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply the quantum Fourier transform to target qubits.

    Textbook construction: for each qubit j, apply H, then controlled-phase
    rotations R_{k-j+1} controlled by every later qubit. Optionally finish
    with SWAPs to reverse the qubit order (so the output is in the same
    endianness as the input).

    :param isa: instruction set architecture
    :param target: list of qubit indices (at least 1)
    :param kwargs:
        do_swaps: if True (default), append bit-reversal SWAPs at the end
        endianness: "little" (default) means target[0] is the least-significant
                    qubit; "big" reverses the iteration order
    :return: list of instructions implementing QFT
    """
    do_swaps = kwargs.get("do_swaps", True)
    endianness = kwargs.get("endianness", "little")
    n = len(target)

    if n < 1:
        raise ValueError("qft requires at least 1 qubit")

    if endianness not in ("little", "big"):
        raise ValueError(
            f"endianness must be 'little' or 'big', got '{endianness}'"
        )

    qubits = list(target) if endianness == "little" else list(reversed(target))
    instructions = []

    # Iterate from MSB (q[n-1]) down to LSB (q[0]). H on the current qubit
    # extracts its bit, then controlled phases from every less-significant
    # qubit add their fractional contributions to the running phase.
    for j in reversed(range(n)):
        instructions.append(isa.h(tg=qubits[j]))
        for k in reversed(range(j)):
            angle = 2.0 * np.pi / (1 << (j - k + 1))
            instructions.append(
                isa.cp(ct=qubits[k], tg=qubits[j], params=[angle])
            )

    if do_swaps:
        for i in range(n // 2):
            instructions.append(
                isa.swap(tg_a=qubits[i], tg_b=qubits[n - 1 - i])
            )

    return instructions


def iqft(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply the inverse quantum Fourier transform to target qubits.

    Constructed as the time-reversal of qft: SWAPs first (if enabled), then
    the H + controlled-phase pattern in reverse order with each phase angle
    negated. Hadamard is self-inverse so its position simply reverses.

    :param isa: instruction set architecture
    :param target: list of qubit indices (at least 1)
    :param kwargs:
        do_swaps: if True (default), prepend bit-reversal SWAPs
        endianness: "little" (default) or "big"
    :return: list of instructions implementing inverse QFT
    """
    do_swaps = kwargs.get("do_swaps", True)
    endianness = kwargs.get("endianness", "little")
    n = len(target)

    if n < 1:
        raise ValueError("iqft requires at least 1 qubit")

    if endianness not in ("little", "big"):
        raise ValueError(
            f"endianness must be 'little' or 'big', got '{endianness}'"
        )

    qubits = list(target) if endianness == "little" else list(reversed(target))
    instructions = []

    if do_swaps:
        for i in range(n // 2):
            instructions.append(
                isa.swap(tg_a=qubits[i], tg_b=qubits[n - 1 - i])
            )

    # Time-reverse of qft's MSB-down loop: iterate j from LSB up to MSB,
    # emit the negated controlled phases in reverse order, then the H.
    for j in range(n):
        for k in range(j):
            angle = -2.0 * np.pi / (1 << (j - k + 1))
            instructions.append(
                isa.cp(ct=qubits[k], tg=qubits[j], params=[angle])
            )
        instructions.append(isa.h(tg=qubits[j]))

    return instructions
