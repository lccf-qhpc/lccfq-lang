"""Single-qubit rotation primitives."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


def rotate(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Apply a rotation gate to each qubit in target.

    :param isa: instruction set architecture
    :param target: list of qubit indices
    :param kwargs:
        axis: "x", "y", or "z" — selects Rx, Ry, or Rz
        theta: float (broadcast to all targets) or list of floats
               (one per target qubit)
    :return: list of rotation instructions
    """
    axis = kwargs["axis"].lower()
    theta = kwargs["theta"]

    if axis not in ("x", "y", "z"):
        raise ValueError(
            f"Axis must be 'x', 'y', or 'z', got '{axis}'"
        )

    gate_name = f"r{axis}"
    gate_fn = getattr(isa, gate_name)
    n = len(target)

    if isinstance(theta, (int, float)):
        angles = [float(theta)] * n
    else:
        angles = list(theta)
        if len(angles) != n:
            raise ValueError(
                f"theta length {len(angles)} != target count {n}"
            )

    return [gate_fn(tg=target[i], params=[angles[i]]) for i in range(n)]
