"""Private decomposition kernels for multi-controlled gates.

Implements two decomposition strategies:

Barenco mode (clean ancilla):
    Uses Barenco et al. (1995) Lemma 7.5 (two-half dirty-ancilla scheme).
    Requires one clean ancilla qubit in |0>.  O(n) depth.

V-chain mode (no ancilla):
    Uses a recursive V-gate ladder.  At each recursion level k,
    C^n(V_{2^k}) is built from C^{n-1}(V_{2^{k+1}}) sub-blocks.

    V_{2^k} = exp(i*pi*X / 2^k) = e^{i*pi/2^k} * Rx(pi/2^{k-1})

    so  C(V_{2^k})(ctrl -> tgt) = P(pi/2^k)(ctrl)
                                   . CRx(pi/2^{k-1})(ctrl -> tgt)

    The recursion for C^n(V_k)(controls, target) with n >= 2 is the
    5-term identity (verified by numerical unitary comparison):

        C^n(V_k) = C^{n-1}(V_{k+1})(lo -> t)
                 . CX(c_hi -> c_pen)
                 . [C^{n-1}(V_{k+1})(lo -> t)]†
                 . CX(c_hi -> c_pen)
                 . C^{n-1}(V_{k+1})(lo[:-1]+[c_hi] -> t)

    where lo = controls[:-1], c_hi = controls[-1], c_pen = lo[-1].

    The outer MCX uses this with k=2 (V_2 = sqrt(X)):

        C^n(X) = C^{n-1}(V_2)(lo -> t)
               . CX(c_hi -> c_pen)
               . [C^{n-1}(V_2)(lo -> t)]†
               . CX(c_hi -> c_pen)
               . C^{n-1}(V_2)(lo[:-1]+[c_hi] -> t)

    Gate cost: O(5^{n-2} * (n-1)) which is approximately O(5^n) — exponential
    in n, but for practical n <= 8 this is acceptable and exactly correct.

All functions assume their arguments have already been validated by the
public API in multicontrol.py.  No **kwargs, no validation here.

Little-endian convention: qubit 0 is the LSB of the state-vector index,
matching _sim.py and _equiv_native.py.
"""
from typing import List, Optional
import math

from ..arch.instruction import Instruction
from ..arch.isa import ISA

# ---------------------------------------------------------------------------
# ISA-level building blocks
# ---------------------------------------------------------------------------

def _toffoli(isa: ISA, c0: int, c1: int, tg: int) -> List[Instruction]:
    """Standard Toffoli (CCX) decomposition into 1q + CX gates.

    Uses the classic T-gate decomposition (6 CX, 7 T/Tdg, 2 H).
    References: Nielsen & Chuang Fig 4.9; Selinger 2013.
    """
    result = [isa.h(tg=tg)]
    result.append(isa.cx(ct=c1, tg=tg))
    result.append(isa.tdg(tg=tg))
    result.append(isa.cx(ct=c0, tg=tg))
    result.append(isa.t(tg=tg))
    result.append(isa.cx(ct=c1, tg=tg))
    result.append(isa.tdg(tg=tg))
    result.append(isa.cx(ct=c0, tg=tg))
    result.append(isa.t(tg=tg))
    result.append(isa.t(tg=c1))
    result.append(isa.h(tg=tg))
    result.append(isa.cx(ct=c0, tg=c1))
    result.append(isa.t(tg=c0))
    result.append(isa.tdg(tg=c1))
    result.append(isa.cx(ct=c0, tg=c1))
    return result


# ---------------------------------------------------------------------------
# V-chain (no ancilla) decomposition
# ---------------------------------------------------------------------------

def _cv_k(isa: ISA, ctrl: int, tg: int, k: int) -> List[Instruction]:
    """C(V_{2^k})(ctrl -> tg) = P(pi/2^k)(ctrl) . CRx(pi/2^{k-1})(ctrl -> tg).

    V_{2^k} = exp(i*pi*X/2^k) = e^{i*pi/2^k} * Rx(pi/2^{k-1}).

    The controlled version adds phase P(pi/2^k) on the control qubit
    (relative to the identity branch) in addition to CRx on the target.
    """
    phase_angle = math.pi / (1 << k)           # pi/2^k
    rx_angle = math.pi / (1 << (k - 1))         # pi/2^{k-1}
    return [
        isa.p(tg=ctrl, params=[phase_angle]),
        isa.crx(ct=ctrl, tg=tg, params=[rx_angle]),
    ]


def _mcv_block(
    isa: ISA,
    controls: List[int],
    target: int,
    k: int,
    dagger: bool = False,
) -> List[Instruction]:
    """C^n(V_{2^k})(controls -> target) using the 5-term V-ladder.

    For n == 1: directly emits P(pi/2^k) + CRx(pi/2^{k-1}).
    For n >= 2: recursive 5-term identity (see module docstring).

    The dagger (adjoint) is implemented by reversing the non-dagger
    instruction list and negating all rotation/phase parameters.
    """
    n = len(controls)
    if n == 1:
        phase_angle = math.pi / (1 << k)
        rx_angle = math.pi / (1 << (k - 1))
        if not dagger:
            return [
                isa.p(tg=controls[0], params=[phase_angle]),
                isa.crx(ct=controls[0], tg=target, params=[rx_angle]),
            ]
        else:
            return [
                isa.crx(ct=controls[0], tg=target, params=[-rx_angle]),
                isa.p(tg=controls[0], params=[-phase_angle]),
            ]

    # n >= 2: 5-term recursion
    c_hi = controls[-1]
    lo = controls[:-1]
    c_pen = lo[-1]

    if not dagger:
        # Application order (left to right):
        # 1. C^{n-1}(V_{k+1})(lo -> t)
        # 2. CX(c_hi -> c_pen)
        # 3. [C^{n-1}(V_{k+1})(lo -> t)]†
        # 4. CX(c_hi -> c_pen)
        # 5. C^{n-1}(V_{k+1})(lo[:-1]+[c_hi] -> t)
        result = []
        result.extend(_mcv_block(isa, lo, target, k + 1, dagger=False))
        result.append(isa.cx(ct=c_hi, tg=c_pen))
        result.extend(_mcv_block(isa, lo, target, k + 1, dagger=True))
        result.append(isa.cx(ct=c_hi, tg=c_pen))
        result.extend(_mcv_block(isa, lo[:-1] + [c_hi], target, k + 1, dagger=False))
        return result
    else:
        # Dagger = reverse order and flip dagger flags:
        # 5†. [C^{n-1}(V_{k+1})(lo[:-1]+[c_hi] -> t)]†
        # 4†. CX(c_hi -> c_pen)
        # 3†. C^{n-1}(V_{k+1})(lo -> t)
        # 2†. CX(c_hi -> c_pen)
        # 1†. [C^{n-1}(V_{k+1})(lo -> t)]†
        result = []
        result.extend(_mcv_block(isa, lo[:-1] + [c_hi], target, k + 1, dagger=True))
        result.append(isa.cx(ct=c_hi, tg=c_pen))
        result.extend(_mcv_block(isa, lo, target, k + 1, dagger=False))
        result.append(isa.cx(ct=c_hi, tg=c_pen))
        result.extend(_mcv_block(isa, lo, target, k + 1, dagger=True))
        return result


def _vchain_mcx_internal(isa: ISA, controls: List[int], target: int) -> List[Instruction]:
    """C^n(X) without ancilla via V-gate ladder (Barenco Corollary 7.4).

    Implements the 5-term recursion using C^{n-1}(V_2) sub-blocks:

        C^n(X)(controls, t) =
            C^{n-1}(V_2)(lo -> t)
          . CX(c_hi -> c_pen)
          . [C^{n-1}(V_2)(lo -> t)]†
          . CX(c_hi -> c_pen)
          . C^{n-1}(V_2)(lo[:-1]+[c_hi] -> t)

    where lo = controls[:-1], c_hi = controls[-1], c_pen = lo[-1] = controls[-2].

    Base cases:
        n=0: X(target)
        n=1: CX(controls[0] -> target)
        n=2: Toffoli (T-gate decomposition)
    """
    n = len(controls)

    if n == 0:
        return [isa.x(tg=target)]
    if n == 1:
        return [isa.cx(ct=controls[0], tg=target)]
    if n == 2:
        return _toffoli(isa, controls[0], controls[1], target)

    # n >= 3: 5-term V_2 recursion
    c_hi = controls[-1]
    lo = controls[:-1]   # n-1 controls
    c_pen = lo[-1]       # controls[-2]

    result = []
    # C^{n-1}(V_2)(lo -> target)
    result.extend(_mcv_block(isa, lo, target, k=2, dagger=False))
    # CX(c_hi -> c_pen)
    result.append(isa.cx(ct=c_hi, tg=c_pen))
    # [C^{n-1}(V_2)(lo -> target)]†
    result.extend(_mcv_block(isa, lo, target, k=2, dagger=True))
    # CX(c_hi -> c_pen)
    result.append(isa.cx(ct=c_hi, tg=c_pen))
    # C^{n-1}(V_2)(lo[:-1]+[c_hi] -> target)
    result.extend(_mcv_block(isa, lo[:-1] + [c_hi], target, k=2, dagger=False))
    return result


# ---------------------------------------------------------------------------
# Barenco clean-ancilla scheme (Lemma 7.5)
# ---------------------------------------------------------------------------

def _barenco_mcx(isa: ISA, controls: List[int], target: int,
                 ancilla: int) -> List[Instruction]:
    """Decompose MCX with n >= 3 controls using one clean ancilla.

    Uses Barenco et al. (1995) Lemma 7.5 (two-half scheme):
      1. MCX(lo → ancilla)  -- put partial AND result in ancilla
      2. MCX(hi + [ancilla] → target)  -- compute final result
      3. MCX(lo → ancilla)  -- restore ancilla to |0>
      4. MCX(hi + [ancilla] → target)  -- reapply to finalize

    For each recursive call, if it has <= 2 controls, use Toffoli directly.
    For longer sub-problems, recurse with further sub-ancillas taken from
    the "dirty" available qubits via the dirty-ancilla trick.
    """
    n = len(controls)

    if n == 1:
        return [isa.cx(ct=controls[0], tg=target)]
    if n == 2:
        return _toffoli(isa, controls[0], controls[1], target)

    # n >= 3: two-half split
    half = (n + 1) // 2
    lo = controls[:half]
    hi = controls[half:]

    def _sub_mcx(ctrls, tgt, dirty_pool):
        """MCX that may use dirty_pool[0] as a dirty ancilla if needed."""
        if len(ctrls) == 1:
            return [isa.cx(ct=ctrls[0], tg=tgt)]
        if len(ctrls) == 2:
            return _toffoli(isa, ctrls[0], ctrls[1], tgt)
        # Need an ancilla — grab from dirty pool
        if not dirty_pool:
            # Fall back to vchain (no clean ancilla available)
            return _vchain_mcx_internal(isa, ctrls, tgt)
        dirty_anc = dirty_pool[0]
        new_dirty = dirty_pool[1:] + [tgt]
        half2 = (len(ctrls) + 1) // 2
        lo2 = ctrls[:half2]
        hi2 = ctrls[half2:]
        instrs = []
        instrs.extend(_sub_mcx(lo2, dirty_anc, hi2 + new_dirty))
        instrs.extend(_sub_mcx(hi2 + [dirty_anc], tgt, lo2 + new_dirty))
        instrs.extend(_sub_mcx(lo2, dirty_anc, hi2 + new_dirty))
        instrs.extend(_sub_mcx(hi2 + [dirty_anc], tgt, lo2 + new_dirty))
        return instrs

    result = []
    result.extend(_sub_mcx(lo, ancilla, hi + [target]))
    result.extend(_sub_mcx(hi + [ancilla], target, lo))
    result.extend(_sub_mcx(lo, ancilla, hi + [target]))
    result.extend(_sub_mcx(hi + [ancilla], target, lo))
    return result


# ---------------------------------------------------------------------------
# Public decomposition functions
# ---------------------------------------------------------------------------

def decompose_mcx(
    isa: ISA,
    controls: List[int],
    target: int,
    mode: str,
    ancilla: Optional[int],
) -> List[Instruction]:
    """Decompose an n-controlled X (n >= 2) into 1q+2q gates.

    Preconditions (assumed, not checked):
      - len(controls) >= 2
      - target not in controls
      - mode in {"barenco", "vchain"}
      - if mode == "barenco": ancilla is not None and ancilla not in controls + [target]
    """
    if mode == "barenco":
        return _barenco_mcx(isa, controls, target, ancilla)
    else:
        return _vchain_mcx_internal(isa, controls, target)


def decompose_mcz(
    isa: ISA,
    controls: List[int],
    target: int,
    mode: str,
    ancilla: Optional[int],
) -> List[Instruction]:
    """MCZ via H-sandwich on target around an MCX.

    MCZ(c_0..c_{n-1}, t) = H_t . MCX(c_0..c_{n-1}, t) . H_t
    """
    result = []
    result.append(isa.h(tg=target))
    result.extend(decompose_mcx(isa, controls, target, mode, ancilla))
    result.append(isa.h(tg=target))
    return result


def decompose_mcry(
    isa: ISA,
    controls: List[int],
    target: int,
    theta: float,
    mode: str,
    ancilla: Optional[int],
) -> List[Instruction]:
    """Multi-controlled R_y(theta) via the conjugation pattern.

    MCRY(theta) = Ry(theta/2) . MCX . Ry(-theta/2) . MCX
    where MCX uses controls and the same mode/ancilla.
    """
    result = []
    result.append(isa.ry(tg=target, params=[theta / 2]))
    result.extend(decompose_mcx(isa, controls, target, mode, ancilla))
    result.append(isa.ry(tg=target, params=[-theta / 2]))
    result.extend(decompose_mcx(isa, controls, target, mode, ancilla))
    return result


def decompose_mcrz(
    isa: ISA,
    controls: List[int],
    target: int,
    theta: float,
    mode: str,
    ancilla: Optional[int],
) -> List[Instruction]:
    """Multi-controlled R_z(theta) via phase-kickback identity.

    MCRZ(theta) = Rz(theta/2) . MCX . Rz(-theta/2) . MCX
    where MCX uses controls and the same mode/ancilla.
    """
    result = []
    result.append(isa.rz(tg=target, params=[theta / 2]))
    result.extend(decompose_mcx(isa, controls, target, mode, ancilla))
    result.append(isa.rz(tg=target, params=[-theta / 2]))
    result.extend(decompose_mcx(isa, controls, target, mode, ancilla))
    return result
