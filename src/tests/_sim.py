"""Dense-vector simulator for the gate subset emitted by lang/* blocks.

Test-only helper. Supports h, x, y, z, s, sdg, t, tdg, p, rx, ry, rz,
cx, cy, cz, crx, cry, crz, cp, swap, plus multi-controlled x and z
(Instructions with control_qubits of length > 1).
Little-endian indexing: qubit 0 is the LSB of state-vector index.
"""
import numpy as np

_ISQRT2 = 1 / np.sqrt(2)

H = _ISQRT2 * np.array([[1, 1], [1, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, (1 + 1j) * _ISQRT2]], dtype=complex)
TDG = np.array([[1, 0], [0, (1 - 1j) * _ISQRT2]], dtype=complex)


def _u_p(theta):
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def _u_rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _u_ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _u_rz(theta):
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )


def _apply_one_qubit(state, q, n, u):
    # Perf #8: vectorized via numpy fancy indexing. For each pair of
    # indices (zero-bit-at-q, one-bit-at-q), apply the 2x2 unitary.
    mask = 1 << q
    indices = np.arange(1 << n)
    zero_idx = indices[(indices & mask) == 0]
    one_idx = zero_idx | mask
    a = state[zero_idx]
    b = state[one_idx]
    new = state.copy()
    new[zero_idx] = u[0, 0] * a + u[0, 1] * b
    new[one_idx] = u[1, 0] * a + u[1, 1] * b
    return new


def _apply_multi_ctrl(state, controls, target, n, u):
    # Perf #8: vectorized. Only indices where all control bits are 1 AND
    # the target bit is 0 are "anchor" indices; the partner is the same
    # index with the target bit flipped. Apply the 2x2 unitary to each
    # (anchor, partner) pair.
    ctrl_mask = 0
    for c in controls:
        ctrl_mask |= 1 << c
    tgt_mask = 1 << target
    indices = np.arange(1 << n)
    active_zero = (
        ((indices & ctrl_mask) == ctrl_mask) & ((indices & tgt_mask) == 0)
    )
    zero_idx = indices[active_zero]
    one_idx = zero_idx | tgt_mask
    a = state[zero_idx]
    b = state[one_idx]
    new = state.copy()
    new[zero_idx] = u[0, 0] * a + u[0, 1] * b
    new[one_idx] = u[1, 0] * a + u[1, 1] * b
    return new


def _apply_swap(state, a, b, n):
    # Perf #8: vectorized. For indices where bit-a != bit-b, take amplitude
    # from the partner (a/b bits flipped); otherwise unchanged.
    indices = np.arange(1 << n)
    bit_a = (indices >> a) & 1
    bit_b = (indices >> b) & 1
    swap_mask = (1 << a) | (1 << b)
    different = bit_a != bit_b
    return np.where(different, state[indices ^ swap_mask], state)


def simulate(instructions, n: int, initial: np.ndarray = None) -> np.ndarray:
    state = (
        np.zeros(1 << n, dtype=complex) if initial is None else initial.copy()
    )
    if initial is None:
        state[0] = 1.0

    for inst in instructions:
        sym = inst.symbol
        tgs = inst.target_qubits
        cts = inst.control_qubits
        ps = inst.params

        if sym == "h":
            state = _apply_one_qubit(state, tgs[0], n, H)
        elif sym == "x":
            if inst.is_controlled and cts:
                state = _apply_multi_ctrl(state, cts, tgs[0], n, X)
            else:
                state = _apply_one_qubit(state, tgs[0], n, X)
        elif sym == "y":
            state = _apply_one_qubit(state, tgs[0], n, Y)
        elif sym == "z":
            if inst.is_controlled and cts:
                state = _apply_multi_ctrl(state, cts, tgs[0], n, Z)
            else:
                state = _apply_one_qubit(state, tgs[0], n, Z)
        elif sym == "s":
            state = _apply_one_qubit(state, tgs[0], n, S)
        elif sym == "sdg":
            state = _apply_one_qubit(state, tgs[0], n, SDG)
        elif sym == "t":
            state = _apply_one_qubit(state, tgs[0], n, T)
        elif sym == "tdg":
            state = _apply_one_qubit(state, tgs[0], n, TDG)
        elif sym == "p":
            state = _apply_one_qubit(state, tgs[0], n, _u_p(ps[0]))
        elif sym == "rx":
            state = _apply_one_qubit(state, tgs[0], n, _u_rx(ps[0]))
        elif sym == "ry":
            state = _apply_one_qubit(state, tgs[0], n, _u_ry(ps[0]))
        elif sym == "rz":
            state = _apply_one_qubit(state, tgs[0], n, _u_rz(ps[0]))
        elif sym == "cp":
            state = _apply_multi_ctrl(state, [cts[0]], tgs[0], n, _u_p(ps[0]))
        elif sym == "cz":
            state = _apply_multi_ctrl(state, [cts[0]], tgs[0], n, Z)
        elif sym == "cx":
            state = _apply_multi_ctrl(state, [cts[0]], tgs[0], n, X)
        elif sym == "cy":
            state = _apply_multi_ctrl(state, [cts[0]], tgs[0], n, Y)
        elif sym == "crx":
            state = _apply_multi_ctrl(state, [cts[0]], tgs[0], n, _u_rx(ps[0]))
        elif sym == "cry":
            state = _apply_multi_ctrl(state, [cts[0]], tgs[0], n, _u_ry(ps[0]))
        elif sym == "crz":
            state = _apply_multi_ctrl(state, [cts[0]], tgs[0], n, _u_rz(ps[0]))
        elif sym == "swap":
            state = _apply_swap(state, cts[0], tgs[0], n)
        else:
            raise NotImplementedError(f"simulate: unsupported gate '{sym}'")

    return state


def basis_state(value: int, n: int) -> np.ndarray:
    s = np.zeros(1 << n, dtype=complex)
    s[value] = 1.0
    return s


# ---------------------------------------------------------------------------
# Pauli operator construction and Hermitian matrix exponentiation
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = X
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = Z


def pauli_op(paulis: dict, n: int) -> np.ndarray:
    """Build a 2**n x 2**n Pauli-string operator under little-endian
    convention (position 0 is LSB, i.e., the rightmost factor of the
    Kronecker product)."""
    table = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    factors = []
    for q in reversed(range(n)):
        p = paulis.get(q, "I").upper()
        factors.append(table[p])
    out = factors[0]
    for f in factors[1:]:
        out = np.kron(out, f)
    return out


def hamiltonian_op(terms, n: int) -> np.ndarray:
    """Build a 2**n x 2**n Hermitian matrix from a list of (coef, paulis) terms."""
    H = np.zeros((1 << n, 1 << n), dtype=complex)
    for coef, paulis in terms:
        H = H + float(coef) * pauli_op(paulis, n)
    return H


def expm_hermitian(H: np.ndarray) -> np.ndarray:
    """Matrix exponential exp(H) for a Hermitian H via eigendecomposition.
    Pass i*H_phys * factor or whatever is needed for the actual unitary;
    this helper just exponentiates the matrix it's given."""
    w, V = np.linalg.eigh(H)
    return V @ np.diag(np.exp(w)) @ V.conj().T


def unitary_of_evolution(H: np.ndarray, t: float) -> np.ndarray:
    """exp(-i t H) for Hermitian H."""
    w, V = np.linalg.eigh(H)
    return V @ np.diag(np.exp(-1j * t * w)) @ V.conj().T
