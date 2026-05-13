"""Quantum error correction primitives: syndrome extraction."""

from typing import Dict, List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def _validate_stabilizer(stabilizer: Dict[int, str], n: int) -> Dict[int, str]:
    """Drop identity entries and validate the rest of a stabilizer spec."""
    out = {}
    for pos, p in stabilizer.items():
        if not isinstance(pos, int) or not 0 <= pos < n:
            raise ValueError(
                f"stabilizer position {pos!r} not in [0, {n})"
            )
        if not isinstance(p, str):
            raise TypeError(
                f"stabilizer value must be a string, got {type(p).__name__}"
            )
        pu = p.upper()
        if pu not in ("I", "X", "Y", "Z"):
            raise ValueError(
                f"stabilizer value must be one of 'I', 'X', 'Y', 'Z'; "
                f"got '{p}'"
            )
        if pu != "I":
            out[pos] = pu
    return out


def syndrome(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Single stabilizer measurement via phase-kickback onto an ancilla.

    Pattern:
        H ancilla -> for each (pos, P) in stabilizer: ctrl-P from ancilla
        to target[pos] -> H ancilla -> (optional) measure ancilla.

    The ancilla returns the +1/-1 eigenvalue of the stabilizer as a 0/1
    classical (or quantum, if measurement is suppressed) bit. Works for any
    Pauli string built from X, Y, Z (identity entries are skipped).

    :param isa: instruction set architecture
    :param target: list of qubit indices the stabilizer acts on (length n >= 1)
    :param kwargs:
        stabilizer: dict {pos: 'I'|'X'|'Y'|'Z'} where pos is a position in
                    target (0 .. n-1). 'I' and missing positions are identity.
        ancilla: int qubit index; must not appear in target and must be in |0>
                 at circuit entry
        measure: bool (default True). If False, the syndrome stays on the
                 ancilla as a coherent bit (|0> for +1 eigenvalue, |1> for -1).
    :return: list of instructions implementing the extraction
    """
    stabilizer = kwargs["stabilizer"]
    ancilla = kwargs["ancilla"]
    measure = kwargs.get("measure", True)
    n = len(target)

    if n < 1:
        raise ValueError("syndrome requires at least 1 target qubit")
    if ancilla in target:
        raise ValueError(
            f"ancilla {ancilla} must not appear in target {target}"
        )

    active = _validate_stabilizer(stabilizer, n)

    gate_for = {"X": "cx", "Y": "cy", "Z": "cz"}

    instructions = [isa.h(tg=ancilla)]
    for pos in sorted(active.keys()):
        gate_fn = getattr(isa, gate_for[active[pos]])
        instructions.append(gate_fn(ct=ancilla, tg=target[pos]))
    instructions.append(isa.h(tg=ancilla))

    if measure:
        instructions.append(isa.measure(tgs=[ancilla]))

    return instructions
