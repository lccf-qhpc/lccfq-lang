"""
Filename: codes_test.py
Author: Santiago Nunez-Corrales
Date: 2026-05-12
Version: 1.0
Description:
    Tests for QEC primitives: syndrome extraction.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.lang.codes import syndrome

from ._sim import simulate, basis_state, pauli_op


@pytest.fixture
def isa():
    return ISA("test")


def symbols(instrs):
    return [i.symbol for i in instrs]


def _pauli_eigenstate(paulis: dict, n: int, eigenvalue: int):
    """Return a normalized eigenstate of the Pauli operator with the given
    eigenvalue (+1 or -1)."""
    op = pauli_op(paulis, n)
    w, V = np.linalg.eigh(op)
    # Eigenvalues are +/-1 for Pauli ops; pick the matching eigenvector.
    idx = int(np.argmin(np.abs(w - eigenvalue)))
    return V[:, idx]


def _strip_measurement(circuit):
    """Return the circuit without a trailing measure instruction (the
    simulator does not implement measurement)."""
    return [i for i in circuit if i.symbol != "measure"]


# ===========================================================================
# syndrome — structural correctness
# ===========================================================================

class TestSyndromeStructure:
    def test_z_stabilizer_single_qubit(self, isa):
        result = syndrome(
            isa, [0], stabilizer={0: "Z"}, ancilla=1, measure=False,
        )
        assert symbols(result) == ["h", "cz", "h"]

    def test_x_stabilizer_uses_cx(self, isa):
        result = syndrome(
            isa, [0], stabilizer={0: "X"}, ancilla=1, measure=False,
        )
        assert symbols(result) == ["h", "cx", "h"]

    def test_y_stabilizer_uses_cy(self, isa):
        result = syndrome(
            isa, [0], stabilizer={0: "Y"}, ancilla=1, measure=False,
        )
        assert symbols(result) == ["h", "cy", "h"]

    def test_multi_qubit_zzz(self, isa):
        result = syndrome(
            isa, [0, 1, 2],
            stabilizer={0: "Z", 1: "Z", 2: "Z"}, ancilla=3, measure=False,
        )
        assert symbols(result) == ["h", "cz", "cz", "cz", "h"]

    def test_mixed_string_xyz(self, isa):
        result = syndrome(
            isa, [0, 1, 2],
            stabilizer={0: "X", 1: "Y", 2: "Z"}, ancilla=3, measure=False,
        )
        assert symbols(result) == ["h", "cx", "cy", "cz", "h"]

    def test_identity_entries_skipped(self, isa):
        result = syndrome(
            isa, [0, 1, 2],
            stabilizer={0: "Z", 1: "I", 2: "Z"}, ancilla=3, measure=False,
        )
        # Only two controlled gates (the I is dropped)
        assert symbols(result) == ["h", "cz", "cz", "h"]

    def test_measurement_appended_by_default(self, isa):
        result = syndrome(
            isa, [0], stabilizer={0: "Z"}, ancilla=1,
        )
        assert result[-1].symbol == "measure"
        assert result[-1].target_qubits == [1]

    def test_no_measurement_when_disabled(self, isa):
        result = syndrome(
            isa, [0], stabilizer={0: "Z"}, ancilla=1, measure=False,
        )
        assert not any(i.symbol == "measure" for i in result)


# ===========================================================================
# syndrome — eigenvalue extraction is correct
# ===========================================================================

class TestSyndromeEigenvalue:
    """For an eigenstate |psi> of stabilizer S with S|psi> = s|psi> (s = +/-1),
    syndrome extraction must leave ancilla in |0> when s=+1 and |1> when s=-1."""

    @pytest.mark.parametrize("paulis,n", [
        ({0: "Z"}, 1),
        ({0: "X"}, 1),
        ({0: "Y"}, 1),
        ({0: "Z", 1: "Z"}, 2),
        ({0: "X", 1: "X"}, 2),
        ({0: "Y", 1: "Y"}, 2),
        ({0: "X", 1: "Z"}, 2),
        ({0: "Z", 1: "X", 2: "Y"}, 3),
    ])
    @pytest.mark.parametrize("eigenvalue", [1, -1])
    def test_eigenvalue_extracted(self, isa, paulis, n, eigenvalue):
        total = n + 1  # +1 ancilla
        ancilla = n
        target = list(range(n))

        # Build an eigenstate on the data qubits, then tensor in |0> ancilla.
        data_state = _pauli_eigenstate(paulis, n, eigenvalue)
        # state-vector convention: ancilla is the next-most-significant bit.
        # Index = ancilla_bit * 2^n + data_index.
        full = np.zeros(1 << total, dtype=complex)
        for data_idx in range(1 << n):
            full[data_idx] = data_state[data_idx]  # ancilla bit = 0

        circuit = _strip_measurement(
            syndrome(isa, target, stabilizer=paulis, ancilla=ancilla)
        )
        out = simulate(circuit, total, full)

        # Marginalize over data: prob of ancilla=0 vs ancilla=1.
        probs = np.abs(out) ** 2
        p0 = sum(probs[idx] for idx in range(1 << total)
                 if ((idx >> ancilla) & 1) == 0)
        p1 = sum(probs[idx] for idx in range(1 << total)
                 if ((idx >> ancilla) & 1) == 1)

        if eigenvalue == 1:
            assert p0 > 1 - 1e-9, (
                f"+1 eigenstate of {paulis} should give ancilla=0; "
                f"p0={p0}, p1={p1}"
            )
        else:
            assert p1 > 1 - 1e-9, (
                f"-1 eigenstate of {paulis} should give ancilla=1; "
                f"p0={p0}, p1={p1}"
            )


# ===========================================================================
# syndrome — data is preserved (non-demolition measurement)
# ===========================================================================

class TestSyndromeNonDemolition:
    """The data state on an eigenstate of S should be left untouched
    (up to overall phase)."""

    def test_data_unchanged_on_z_eigenstate(self, isa):
        # +1 eigenstate of Z0Z1: |00> + |11> over sqrt(2).
        ancilla = 2
        target = [0, 1]
        data = (basis_state(0, 2) + basis_state(3, 2)) / np.sqrt(2)
        # tensor with |0> ancilla
        full = np.zeros(8, dtype=complex)
        for idx in range(4):
            full[idx] = data[idx]

        circuit = _strip_measurement(
            syndrome(isa, target, stabilizer={0: "Z", 1: "Z"}, ancilla=ancilla)
        )
        out = simulate(circuit, 3, full)
        # After: ancilla=0 (eigenvalue +1), data unchanged.
        # Marginalize on ancilla=0 sector.
        data_after = np.array([out[idx] for idx in range(4)])
        # Compare to data (up to global phase).
        # Normalize first.
        inner = np.vdot(data, data_after)
        assert np.isclose(np.abs(inner), 1.0, atol=1e-9), (
            f"data state not preserved; |<data|after>| = {np.abs(inner)}"
        )


# ===========================================================================
# syndrome — validation
# ===========================================================================

class TestSyndromeValidation:
    def test_empty_target(self, isa):
        with pytest.raises(ValueError, match="at least 1"):
            syndrome(isa, [], stabilizer={}, ancilla=0)

    def test_ancilla_in_target(self, isa):
        with pytest.raises(ValueError, match="must not appear"):
            syndrome(isa, [0, 1], stabilizer={0: "Z"}, ancilla=1)

    def test_bad_position(self, isa):
        with pytest.raises(ValueError, match="not in"):
            syndrome(isa, [0, 1], stabilizer={5: "X"}, ancilla=2)

    def test_bad_pauli_value(self, isa):
        with pytest.raises(ValueError, match="must be one of"):
            syndrome(isa, [0], stabilizer={0: "W"}, ancilla=1)

    def test_bad_pauli_type(self, isa):
        with pytest.raises(TypeError):
            syndrome(isa, [0], stabilizer={0: 1}, ancilla=1)
