"""Oracle primitives: bit-flip and phase oracles."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA
from .multicontrol import mcx, mcz


def _marked_inputs(predicate, n: int) -> List[int]:
    """Resolve `predicate` (callable, list of ints, or list of bitstrings) into
    a sorted, deduplicated list of marked input integers in [0, 2**n)."""
    dim = 1 << n

    if callable(predicate):
        marked = [x for x in range(dim) if predicate(x)]
    else:
        marked = []
        for item in predicate:
            if isinstance(item, str):
                if len(item) != n or not all(c in "01" for c in item):
                    raise ValueError(
                        f"Bitstring '{item}' must be length {n} of '0'/'1'"
                    )
                # Bitstring is read with item[0] as the most-significant character
                # (textbook convention). Endianness handling happens in the caller.
                marked.append(int(item, 2))
            elif isinstance(item, int):
                if not 0 <= item < dim:
                    raise ValueError(
                        f"Marked input {item} out of range [0, {dim})"
                    )
                marked.append(item)
            else:
                raise TypeError(
                    f"Predicate items must be str or int, got {type(item).__name__}"
                )

    return sorted(set(marked))


def _bit_at(value: int, position: int, n: int, endianness: str) -> int:
    """Return the bit of ``value`` corresponding to target index ``position``.

    With "little" endianness, target[0] is the least-significant bit, so the
    bit at position p is (value >> p) & 1. With "big", target[0] is the most
    significant bit.
    """
    if endianness == "little":
        return (value >> position) & 1
    return (value >> (n - 1 - position)) & 1


def oracle(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Bit-flip oracle: |x>|y> -> |x>|y XOR f(x)>.

    For each marked input x, surround a multi-controlled X (control = target,
    target = ancilla) by single-qubit X gates on the input qubits where the
    corresponding bit of x is 0, so the controls only fire when the input
    matches x.

    :param isa: instruction set architecture
    :param target: list of input-register qubit indices (length n >= 1)
    :param kwargs:
        ancilla: int, the output qubit (must not be in target)
        predicate: callable(int) -> bool, OR a list of marked inputs given as
                   ints in [0, 2**n) or bitstrings of length n
        endianness: "little" (default; target[0] is LSB) or "big"
        scratch_ancilla: int | None = None
            Free scratch qubit forwarded to mcx. Required when mc_mode="barenco".
            Must not appear in target and must not equal ancilla.
        mc_mode: Literal["barenco", "vchain"] = "vchain"
            Decomposition mode for the multi-controlled X. Defaults to "vchain"
            (no scratch needed) for backward compatibility with existing callers.
    :return: list of instructions implementing the bit-flip oracle
    """
    ancilla = kwargs["ancilla"]
    predicate = kwargs["predicate"]
    endianness = kwargs.get("endianness", "little")
    scratch_ancilla = kwargs.get("scratch_ancilla", None)
    mc_mode = kwargs.get("mc_mode", "vchain")
    n = len(target)

    if n < 1:
        raise ValueError("oracle requires at least 1 input qubit")
    if endianness not in ("little", "big"):
        raise ValueError(
            f"endianness must be 'little' or 'big', got '{endianness}'"
        )
    if ancilla in target:
        raise ValueError(
            f"ancilla {ancilla} must not appear in target {target}"
        )
    if scratch_ancilla is not None:
        if scratch_ancilla == ancilla:
            raise ValueError(
                f"scratch_ancilla {scratch_ancilla} must differ from output ancilla {ancilla}"
            )
        if scratch_ancilla in target:
            raise ValueError(
                f"scratch_ancilla {scratch_ancilla} must not appear in target {target}"
            )

    marked = _marked_inputs(predicate, n)
    instructions = []

    for x in marked:
        flips = [
            target[p] for p in range(n)
            if _bit_at(x, p, n, endianness) == 0
        ]
        for q in flips:
            instructions.append(isa.x(tg=q))

        instructions.extend(mcx(
            isa,
            list(target),
            tg=ancilla,
            mode=mc_mode,
            ancilla=scratch_ancilla,
        ))

        for q in flips:
            instructions.append(isa.x(tg=q))

    return instructions


def phase_oracle(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Phase-kickback oracle: |x> -> (-1)^f(x) |x>.

    Ancilla-free by default. For each marked input x, flip the input qubits
    whose corresponding bits are 0 to map |x> -> |11..1>, apply a
    multi-controlled Z (which phase-flips |11..1>), then undo the X flips.

    :param isa: instruction set architecture
    :param target: list of qubit indices (length n >= 1)
    :param kwargs:
        predicate: callable(int) -> bool, OR a list of marked inputs as
                   ints in [0, 2**n) or bitstrings of length n
        endianness: "little" (default; target[0] is LSB) or "big"
        workspace: int | None = None
            Optional clean ancilla forwarded to mcz. Required when
            mc_mode="barenco". Must not appear in target.
        mc_mode: Literal["barenco", "vchain"] = "vchain"
            Decomposition mode. Defaults to "vchain" (no ancilla) for
            backward compatibility.
    :return: list of instructions implementing the phase oracle
    """
    predicate = kwargs["predicate"]
    endianness = kwargs.get("endianness", "little")
    workspace = kwargs.get("workspace", None)
    mc_mode = kwargs.get("mc_mode", "vchain")
    n = len(target)

    if n < 1:
        raise ValueError("phase_oracle requires at least 1 input qubit")
    if endianness not in ("little", "big"):
        raise ValueError(
            f"endianness must be 'little' or 'big', got '{endianness}'"
        )
    if workspace is not None and workspace in target:
        raise ValueError(
            f"workspace qubit {workspace} must not appear in target {target}"
        )

    marked = _marked_inputs(predicate, n)
    instructions = []

    for x in marked:
        flips = [
            target[p] for p in range(n)
            if _bit_at(x, p, n, endianness) == 0
        ]
        for q in flips:
            instructions.append(isa.x(tg=q))

        instructions.extend(mcz(
            isa,
            list(target[:-1]),
            tg=target[-1],
            mode=mc_mode,
            ancilla=workspace,
        ))

        for q in flips:
            instructions.append(isa.x(tg=q))

    return instructions
