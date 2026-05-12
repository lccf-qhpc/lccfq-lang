"""Shared helpers for the lang.* family modules."""

from typing import List

from ..arch.instruction import Instruction
from ..arch.isa import ISA


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
