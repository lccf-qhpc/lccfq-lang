"""Public multi-controlled gate block functions.

Provides mcx, mcz, mcry, mcrz — each conforming to the BlockFactory
contract (isa, target, **kwargs) -> List[Instruction].

Convention: ``target`` is the list of *control* qubit indices.
The actual gate target qubit is passed as the ``tg`` kwarg.

All four functions decompose into 1q + 2q gates only.  Every returned
Instruction satisfies:
    inst.control_qubits is None or len(inst.control_qubits) <= 1
"""
from typing import List, Optional

from ..arch.instruction import Instruction
from ..arch.isa import ISA
from ._mc_decompose import (
    decompose_mcx,
    decompose_mcz,
    decompose_mcry,
    decompose_mcrz,
)

_VALID_MODES = {"barenco", "vchain"}


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

def _validate_common(fname: str, controls, tg: int, mode: str,
                     ancilla: Optional[int]) -> None:
    """Validate arguments shared by all four public functions.

    Raises TypeError or ValueError with the exact messages mandated by §6
    of the integration spec.
    """
    # mode is always required
    if mode is None:
        raise ValueError(
            f"{fname}: 'mode' is required ('barenco' or 'vchain')"
        )
    if mode not in _VALID_MODES:
        raise ValueError(
            f"{fname}: mode must be 'barenco' or 'vchain', got {repr(mode)}"
        )

    # duplicate controls
    seen = set()
    for q in controls:
        if q in seen:
            raise ValueError(f"{fname}: duplicate control qubit {q}")
        seen.add(q)

    # tg must not appear in controls
    if tg in seen:
        raise ValueError(f"{fname}: target qubit {tg} appears in controls")

    # ancilla overlap check (if ancilla provided)
    if ancilla is not None:
        if ancilla == tg or ancilla in seen:
            raise ValueError(
                f"{fname}: ancilla qubit {ancilla} overlaps target/controls"
            )

    # barenco mode with n >= 2 requires ancilla
    n = len(controls)
    if mode == "barenco" and n >= 2 and ancilla is None:
        raise ValueError(f"{fname}: barenco mode requires 'ancilla'")


def _validate_theta(fname: str, theta) -> None:
    """Validate that theta is a real numeric value."""
    if not isinstance(theta, (int, float)):
        raise TypeError(
            f"{fname}: 'theta' must be a real number, got {type(theta).__name__}"
        )


# ---------------------------------------------------------------------------
# mcx
# ---------------------------------------------------------------------------

def mcx(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Multi-controlled X gate decomposed into 1q + 2q native instructions.

    :param isa:    Instruction set architecture.
    :param target: List[int] of control-qubit indices. May be empty (n=0).
    :param kwargs:
        tg:      int (required). Target qubit index.
        mode:    Literal["barenco", "vchain"] (required).
        ancilla: int | None. Required when mode="barenco" and n >= 2.

    :return: List[Instruction] with len(control_qubits) <= 1 for every item.

    :raises TypeError:  missing required ``tg``.
    :raises ValueError: missing/invalid mode; missing ancilla in barenco
                        mode; ancilla overlaps target/tg; tg in target;
                        duplicate control qubit.
    """
    if "tg" not in kwargs:
        raise TypeError("mcx: 'tg' is required")
    tg: int = kwargs["tg"]
    mode: Optional[str] = kwargs.get("mode", None)
    ancilla: Optional[int] = kwargs.get("ancilla", None)

    controls = list(target)
    n = len(controls)

    _validate_common("mcx", controls, tg, mode, ancilla)

    if n == 0:
        return [isa.x(tg=tg)]
    if n == 1:
        return [isa.cx(ct=controls[0], tg=tg)]

    # n >= 2
    return decompose_mcx(isa, controls, tg, mode, ancilla)


# ---------------------------------------------------------------------------
# mcz
# ---------------------------------------------------------------------------

def mcz(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Multi-controlled Z gate decomposed into 1q + 2q native instructions.

    :param isa:    Instruction set architecture.
    :param target: List[int] of control-qubit indices.
    :param kwargs:
        tg:      int (required). Target qubit index.
        mode:    Literal["barenco", "vchain"] (required).
        ancilla: int | None. Required when mode="barenco" and n >= 2.

    :return: List[Instruction] with len(control_qubits) <= 1 for every item.
    """
    if "tg" not in kwargs:
        raise TypeError("mcz: 'tg' is required")
    tg: int = kwargs["tg"]
    mode: Optional[str] = kwargs.get("mode", None)
    ancilla: Optional[int] = kwargs.get("ancilla", None)

    controls = list(target)
    n = len(controls)

    _validate_common("mcz", controls, tg, mode, ancilla)

    if n == 0:
        return [isa.z(tg=tg)]
    if n == 1:
        return [isa.cz(ct=controls[0], tg=tg)]

    # n >= 2
    return decompose_mcz(isa, controls, tg, mode, ancilla)


# ---------------------------------------------------------------------------
# mcry
# ---------------------------------------------------------------------------

def mcry(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Multi-controlled R_y(theta) decomposed into 1q + 2q native instructions.

    :param isa:    Instruction set architecture.
    :param target: List[int] of control-qubit indices.
    :param kwargs:
        tg:      int (required). Target qubit index.
        mode:    Literal["barenco", "vchain"] (required).
        ancilla: int | None. Required when mode="barenco" and n >= 2.
        theta:   float (required). Rotation angle in radians.

    :return: List[Instruction] with len(control_qubits) <= 1 for every item.
    """
    if "tg" not in kwargs:
        raise TypeError("mcry: 'tg' is required")
    if "theta" not in kwargs:
        raise TypeError("mcry: 'theta' is required")
    tg: int = kwargs["tg"]
    theta = kwargs["theta"]
    mode: Optional[str] = kwargs.get("mode", None)
    ancilla: Optional[int] = kwargs.get("ancilla", None)

    _validate_theta("mcry", theta)
    theta = float(theta)

    controls = list(target)
    n = len(controls)

    _validate_common("mcry", controls, tg, mode, ancilla)

    if n == 0:
        return [isa.ry(tg=tg, params=[theta])]
    if n == 1:
        return [isa.cry(ct=controls[0], tg=tg, params=[theta])]

    # n >= 2
    return decompose_mcry(isa, controls, tg, theta, mode, ancilla)


# ---------------------------------------------------------------------------
# mcrz
# ---------------------------------------------------------------------------

def mcrz(isa: ISA, target, **kwargs) -> List[Instruction]:
    """Multi-controlled R_z(theta) decomposed into 1q + 2q native instructions.

    :param isa:    Instruction set architecture.
    :param target: List[int] of control-qubit indices.
    :param kwargs:
        tg:      int (required). Target qubit index.
        mode:    Literal["barenco", "vchain"] (required).
        ancilla: int | None. Required when mode="barenco" and n >= 2.
        theta:   float (required). Rotation angle in radians.

    :return: List[Instruction] with len(control_qubits) <= 1 for every item.
    """
    if "tg" not in kwargs:
        raise TypeError("mcrz: 'tg' is required")
    if "theta" not in kwargs:
        raise TypeError("mcrz: 'theta' is required")
    tg: int = kwargs["tg"]
    theta = kwargs["theta"]
    mode: Optional[str] = kwargs.get("mode", None)
    ancilla: Optional[int] = kwargs.get("ancilla", None)

    _validate_theta("mcrz", theta)
    theta = float(theta)

    controls = list(target)
    n = len(controls)

    _validate_common("mcrz", controls, tg, mode, ancilla)

    if n == 0:
        return [isa.rz(tg=tg, params=[theta])]
    if n == 1:
        return [isa.crz(ct=controls[0], tg=tg, params=[theta])]

    # n >= 2
    return decompose_mcrz(isa, controls, tg, theta, mode, ancilla)
