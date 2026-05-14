"""
Filename: test_peephole_mach.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Unit and semantic tests for the three peephole_mach passes:
    RemoveIdentityMach, MergeAdjacent1Q, EulerXYRecompose.
    Also tests the _xyx_decompose helper for round-trip fidelity.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import math
import numpy as np
import pytest
from lccfq_lang.mach.ir import Gate, Control, Test
from lccfq_lang.opt.builtin.peephole_mach import (
    MergeAdjacent1Q,
    RemoveIdentityMach,
    EulerXYRecompose,
    _xyx_decompose,
)
from lccfq_lang.opt.pass_base import PassContext
from tests.opt._equiv_native import assert_equivalent_native, _u_rx, _u_ry


# Use a stub ISA: passes accept it but only RemoveIdentityMach reads it.
class _StubISA:
    pass


def _ctx():
    return PassContext()


# ---- Gate IR extension: backward compatibility ----

def test_gate_default_tags_empty():
    g = Gate("rx", [0], None, [0.5])
    assert g.tags == {}


def test_gate_default_duration_none():
    g = Gate("rx", [0], None, [0.5])
    assert g.duration is None


def test_gate_tags_not_shared():
    """Each Gate must have its own tags dict (mutable default arg bug check)."""
    g1 = Gate("rx", [0], None, [0.5])
    g2 = Gate("ry", [1], None, [0.3])
    g1.tags["layer"] = 0
    assert "layer" not in g2.tags


def test_gate_to_json_no_extra_keys_by_default():
    g = Gate("rx", [0], None, [0.5])
    j = g.to_json()
    assert set(j.keys()) == {"symbol", "target_qubits", "control_qubits", "params"}


def test_gate_to_json_includes_tags_when_set():
    g = Gate("rx", [0], None, [0.5])
    g.tags["layer"] = 1
    j = g.to_json()
    assert "tags" in j
    assert j["tags"] == {"layer": 1}


def test_gate_to_json_includes_duration_when_set():
    g = Gate("rx", [0], None, [0.5], duration=10.0)
    j = g.to_json()
    assert "duration" in j
    assert j["duration"] == 10.0


# ---- RemoveIdentityMach ----

def test_remove_identity_drops_nop():
    p = [Gate("nop", [], None, [])]
    out = RemoveIdentityMach(_StubISA()).run(p, _ctx())
    assert out == []


def test_remove_identity_drops_zero_rx():
    p = [Gate("rx", [0], None, [0.0])]
    out = RemoveIdentityMach(_StubISA()).run(p, _ctx())
    assert out == []


def test_remove_identity_drops_zero_ry():
    p = [Gate("ry", [0], None, [0.0])]
    out = RemoveIdentityMach(_StubISA()).run(p, _ctx())
    assert out == []


def test_remove_identity_drops_2pi_rx():
    """An rx(2*pi) is effectively zero angle modulo 2*pi."""
    p = [Gate("rx", [0], None, [2 * math.pi])]
    out = RemoveIdentityMach(_StubISA()).run(p, _ctx())
    assert out == []


def test_remove_identity_keeps_nonzero_rx():
    p = [Gate("rx", [0], None, [0.5])]
    out = RemoveIdentityMach(_StubISA()).run(p, _ctx())
    assert len(out) == 1


def test_remove_identity_keeps_zero_sqiswap_unaffected():
    # sqiswap is not parametric; never identity here.
    p = [Gate("sqiswap", [0], [1], [])]
    out = RemoveIdentityMach(_StubISA()).run(p, _ctx())
    assert len(out) == 1


def test_remove_identity_passes_through_classical_ops():
    control = Control("shot", [100])
    p = [control]
    out = RemoveIdentityMach(_StubISA()).run(p, _ctx())
    assert out == [control]


def test_remove_identity_mixed_program():
    p = [
        Gate("nop", [], None, []),
        Gate("rx", [0], None, [0.5]),
        Gate("rx", [1], None, [0.0]),
        Gate("sqiswap", [0], [1], []),
    ]
    out = RemoveIdentityMach(_StubISA()).run(p, _ctx())
    assert len(out) == 2
    assert out[0].symbol == "rx" and out[0].params[0] == 0.5
    assert out[1].symbol == "sqiswap"


# ---- MergeAdjacent1Q ----

def test_merge_adjacent_rx_pair():
    p = [
        Gate("rx", [0], None, [0.5]),
        Gate("rx", [0], None, [0.3]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert len(out) == 1
    assert out[0].symbol == "rx"
    assert math.isclose(out[0].params[0], 0.8)


def test_merge_adjacent_ry_pair():
    p = [
        Gate("ry", [0], None, [0.7]),
        Gate("ry", [0], None, [0.2]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert len(out) == 1
    assert out[0].symbol == "ry"
    assert math.isclose(out[0].params[0], 0.9, abs_tol=1e-9)


def test_merge_adjacent_rx_to_zero_drops_both():
    p = [
        Gate("rx", [0], None, [0.5]),
        Gate("rx", [0], None, [-0.5]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert out == []


def test_merge_does_not_cross_axis():
    p = [
        Gate("rx", [0], None, [0.5]),
        Gate("ry", [0], None, [0.5]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert len(out) == 2


def test_merge_does_not_cross_qubit():
    p = [
        Gate("rx", [0], None, [0.5]),
        Gate("rx", [1], None, [0.5]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert len(out) == 2


def test_merge_blocked_by_intervening_op():
    # An op on the same qubit between two rx prevents merge.
    p = [
        Gate("rx", [0], None, [0.5]),
        Gate("ry", [0], None, [0.1]),
        Gate("rx", [0], None, [0.3]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert len(out) == 3


def test_merge_blocked_by_two_qubit_gate():
    # sqiswap touches qubit 0, blocking the merge.
    p = [
        Gate("rx", [0], None, [0.5]),
        Gate("sqiswap", [0], [1], []),
        Gate("rx", [0], None, [0.3]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert len(out) == 3


def test_merge_semantic_equivalence():
    p = [
        Gate("rx", [0], None, [0.7]),
        Gate("rx", [0], None, [-0.2]),
        Gate("ry", [0], None, [0.3]),
        Gate("ry", [0], None, [0.4]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert_equivalent_native(p, out, n_qubits=1)


def test_merge_chain_of_three_rx():
    """Three consecutive rx on the same qubit should produce one merged gate."""
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("rx", [0], None, [0.2]),
        Gate("rx", [0], None, [0.3]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    # After one pass: first two merge to rx(0.3), then rx(0.3)+rx(0.3)=rx(0.6)
    # Actually: in one pass the pass is forward. first two merge -> rx(0.3),
    # then that merged op vs rx(0.3) -> rx(0.6).
    assert len(out) == 1
    assert math.isclose(out[0].params[0], 0.6, abs_tol=1e-9)
    assert_equivalent_native(p, out, n_qubits=1)


def test_merge_parallel_qubits_independent():
    """Merges on qubit 0 and qubit 1 happen independently."""
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("rx", [1], None, [0.2]),
        Gate("rx", [0], None, [0.3]),
        Gate("rx", [1], None, [0.4]),
    ]
    out = MergeAdjacent1Q(_StubISA()).run(p, _ctx())
    assert len(out) == 2
    symbols = [op.symbol for op in out]
    assert symbols == ["rx", "rx"]
    assert_equivalent_native(p, out, n_qubits=2)


# ---- EulerXYRecompose: _xyx_decompose round-trip ----

def test_xyx_decompose_round_trip():
    rng = np.random.default_rng(0xC0FFEE)
    for _ in range(200):
        a0 = rng.uniform(-np.pi, np.pi)
        b0 = rng.uniform(0, np.pi)
        c0 = rng.uniform(-np.pi, np.pi)
        U = _u_ry(c0) @ _u_rx(b0) @ _u_ry(a0)
        a, b, c = _xyx_decompose(U)
        U_rt = _u_ry(c) @ _u_rx(b) @ _u_ry(a)
        idx = int(np.argmax(np.abs(U)))
        if abs(U.flat[idx]) < 1e-9:
            # Zero U00: pick any non-zero
            idx = int(np.argmax(np.abs(U_rt)))
        if abs(U.flat[idx]) < 1e-9:
            # Both zero: fidelity check
            fid = abs(np.vdot(U_rt.ravel(), U.ravel()))
            assert fid > 0.99
            continue
        ratio = U_rt.flat[idx] / U.flat[idx]
        ratio /= abs(ratio)
        assert np.max(np.abs(U_rt - ratio * U)) < 1e-7


def test_xyx_decompose_identity():
    U = np.eye(2, dtype=complex)
    a, b, c = _xyx_decompose(U)
    U_rt = _u_ry(c) @ _u_rx(b) @ _u_ry(a)
    assert np.max(np.abs(U_rt - U)) < 1e-9


# ---- EulerXYRecompose pass ----

def test_euler_recompose_only_fires_on_runs_geq_4():
    # length 3 must be left alone
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("ry", [0], None, [0.2]),
        Gate("rx", [0], None, [0.3]),
    ]
    out = EulerXYRecompose(_StubISA()).run(p, _ctx())
    assert len(out) == 3  # unchanged


def test_euler_recompose_reduces_5_to_at_most_3():
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("ry", [0], None, [0.2]),
        Gate("rx", [0], None, [0.3]),
        Gate("ry", [0], None, [0.4]),
        Gate("rx", [0], None, [0.5]),
    ]
    out = EulerXYRecompose(_StubISA()).run(p, _ctx())
    assert 1 <= len(out) <= 3
    assert_equivalent_native(p, out, n_qubits=1)


def test_euler_recompose_skips_when_not_reducing():
    # 4 rotations; output count must be <= input count.
    p = [
        Gate("ry", [0], None, [0.1]),
        Gate("rx", [0], None, [0.2]),
        Gate("ry", [0], None, [0.3]),
        Gate("rx", [0], None, [0.4]),
    ]
    out = EulerXYRecompose(_StubISA()).run(p, _ctx())
    assert len(out) <= 4
    assert_equivalent_native(p, out, n_qubits=1)


def test_euler_recompose_does_not_touch_different_qubit():
    # Run on qubit 0 and run on qubit 1 are separate.
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("ry", [0], None, [0.2]),
        Gate("rx", [0], None, [0.3]),
        Gate("ry", [0], None, [0.4]),
        Gate("rx", [1], None, [0.5]),
        Gate("ry", [1], None, [0.6]),
        Gate("rx", [1], None, [0.7]),
        Gate("ry", [1], None, [0.8]),
    ]
    out = EulerXYRecompose(_StubISA()).run(p, _ctx())
    assert len(out) <= 6  # both runs reduced
    assert_equivalent_native(p, out, n_qubits=2)


def test_euler_recompose_broken_by_two_qubit_op():
    # sqiswap breaks the run on qubit 0 into two separate runs.
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("ry", [0], None, [0.2]),
        Gate("sqiswap", [0], [1], []),
        Gate("rx", [0], None, [0.3]),
        Gate("ry", [0], None, [0.4]),
    ]
    out = EulerXYRecompose(_StubISA()).run(p, _ctx())
    # Neither sub-run has length >=4, so all 5 gates pass through.
    assert len(out) == 5


def test_euler_recompose_semantic_6_gate_run():
    """6 mixed rx/ry ops on same qubit: equivalence must hold."""
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("ry", [0], None, [0.2]),
        Gate("rx", [0], None, [0.3]),
        Gate("ry", [0], None, [0.4]),
        Gate("rx", [0], None, [0.5]),
        Gate("ry", [0], None, [0.6]),
    ]
    out = EulerXYRecompose(_StubISA()).run(p, _ctx())
    assert len(out) <= 3
    assert_equivalent_native(p, out, n_qubits=1)
