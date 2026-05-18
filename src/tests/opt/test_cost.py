"""
Filename: test_cost.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for Cost.measure and Cost.scalarize.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import math
import pytest
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.mach.ir import Gate
from lccfq_lang.opt import Cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def instr(symbol, target_qubits=None, control_qubits=None, **kw):
    return Instruction(symbol=symbol, target_qubits=target_qubits,
                       control_qubits=control_qubits, **kw)


# ---------------------------------------------------------------------------
# Empty program
# ---------------------------------------------------------------------------

def test_empty_program_arch():
    c = Cost.measure([], "arch")
    assert c == Cost(0, 0, 0, 0, None)


def test_empty_program_mach():
    c = Cost.measure([], "mach")
    assert c == Cost(0, 0, 0, 0, None)


def test_empty_program_with_calibration_returns_perfect_fidelity():
    """Spec edge case: empty program + qpu_config.calibration → estimated_error=1.0."""
    class DummyCfg:
        calibration = {"per_gate_error": {"x": 0.01}}

    c = Cost.measure([], "arch", qpu_config=DummyCfg())
    assert c == Cost(0, 0, 0, 0, 1.0)


def test_empty_program_with_qpuconfig_no_calibration_returns_none_error():
    """Empty program + qpu_config without calibration → estimated_error=None."""
    class DummyCfg:
        native_2q = ("sqiswap",)
        # no calibration attribute

    c = Cost.measure([], "mach", qpu_config=DummyCfg())
    assert c == Cost(0, 0, 0, 0, None)


def test_estimated_error_clamps_at_0_999999():
    """Spec: per-gate error is clamped to min(e_op, 0.999999) before log."""
    import math
    class DummyCfg:
        # err = 1.0 would otherwise produce log(0) = -inf
        calibration = {"per_gate_error": {"x": 1.0}}

    op = instr("x", target_qubits=[0])
    c = Cost.measure([op], "arch", qpu_config=DummyCfg())
    # Survival per gate is (1 - 0.999999) = 1e-6
    assert c.estimated_error == pytest.approx(1.0 - 0.999999, abs=1e-12)
    assert math.isfinite(c.estimated_error)


# ---------------------------------------------------------------------------
# 4-op hand-built fixture
#   ops: 1q gate, 2q gate, 2q gate, measure
#   linear circuit on qubits 0,1 so depth = 3
#       q0: h - cx - measure
#       q1:   \ cx /
#   Actually:
#       h(q0), cx(ctrl=q0,tg=q1), cx(ctrl=q0,tg=q1), measure(q0)
#   DAG edges: h->cx1 (q0), cx1->cx2 (q0,q1), cx2->measure (q0)
#   Longest path: 3 edges → depth = 4
#   But let us build a simpler fixture where depth is easy to count:
#
#   p0: h(q0)         — 1q
#   p1: cx(q0,q1)     — 2q
#   p2: cx(q1,q2)     — 2q  (depends on p1 via q1)
#   p3: measure(q0)   — measurement (contributes to depth via q0->p1->p3? no, p3 shares q0 with p1 only)
#
#   Qubit flow:
#     q0: p0 -> p1 -> p3
#     q1: p1 -> p2
#     q2: p2
#   Longest path: p0->p1->p2 = 2 edges → depth = 3
#   p3 is a leaf from p1 on q0, path length = 2 edges → depth = 3 as well (same)
# ---------------------------------------------------------------------------

@pytest.fixture
def prog4():
    p0 = instr("h",       target_qubits=[0])
    p1 = instr("cx",      target_qubits=[1], control_qubits=[0], is_controlled=True)
    p2 = instr("cx",      target_qubits=[2], control_qubits=[1], is_controlled=True)
    p3 = instr("measure", target_qubits=[0])
    return [p0, p1, p2, p3]


def test_count_1q(prog4):
    c = Cost.measure(prog4, "arch")
    assert c.count_1q == 1   # only h


def test_count_2q(prog4):
    c = Cost.measure(prog4, "arch")
    assert c.count_2q == 2   # two cx gates


def test_depth(prog4):
    c = Cost.measure(prog4, "arch")
    # Longest qubit-flow path: p0->p1->p2 (2 edges) → depth 3
    assert c.depth == 3


def test_measurement_not_counted(prog4):
    c = Cost.measure(prog4, "arch")
    # measure contributes to neither count_1q nor count_2q
    assert c.count_1q == 1
    assert c.count_2q == 2


# ---------------------------------------------------------------------------
# count_native_2q
# ---------------------------------------------------------------------------

def test_count_native_2q_arch_always_zero(prog4):
    c = Cost.measure(prog4, "arch")
    assert c.count_native_2q == 0


def test_count_native_2q_mach_no_config():
    """With kind=mach and no qpu_config, native_2q falls back to count_2q."""
    p0 = Gate(symbol="cx",      target_qubits=[1], control_qubits=[0], params=[])
    p1 = Gate(symbol="sqiswap", target_qubits=[1], control_qubits=[0], params=[])
    c = Cost.measure([p0, p1], "mach", qpu_config=None)
    assert c.count_native_2q == c.count_2q


def test_count_native_2q_mach_empty_native_set():
    """qpu_config present but native_2q is empty → fall back to count_2q."""
    class DummyCfg:
        native_2q = ()

    p0 = Gate(symbol="cx",      target_qubits=[1], control_qubits=[0], params=[])
    p1 = Gate(symbol="sqiswap", target_qubits=[1], control_qubits=[0], params=[])
    c = Cost.measure([p0, p1], "mach", qpu_config=DummyCfg())
    assert c.count_native_2q == c.count_2q


def test_count_native_2q_mach_with_native_set():
    """Only ops whose symbol is in native_2q are counted."""
    class DummyCfg:
        native_2q = ("sqiswap",)

    p0 = Gate(symbol="cx",      target_qubits=[1], control_qubits=[0], params=[])
    p1 = Gate(symbol="sqiswap", target_qubits=[1], control_qubits=[0], params=[])
    c = Cost.measure([p0, p1], "mach", qpu_config=DummyCfg())
    assert c.count_2q == 2
    assert c.count_native_2q == 1  # only sqiswap


# ---------------------------------------------------------------------------
# scalarize
# ---------------------------------------------------------------------------

def test_scalarize_default_weights(prog4):
    from lccfq_lang.opt.cost import DEFAULT_WEIGHTS
    c = Cost.measure(prog4, "arch")
    expected = (
        DEFAULT_WEIGHTS["depth"]           * c.depth
        + DEFAULT_WEIGHTS["count_1q"]      * c.count_1q
        + DEFAULT_WEIGHTS["count_2q"]      * c.count_2q
        + DEFAULT_WEIGHTS["count_native_2q"] * c.count_native_2q
        # estimated_error is None → no error term
    )
    assert c.scalarize() == pytest.approx(expected)


def test_scalarize_custom_weights():
    c = Cost(depth=2, count_1q=3, count_2q=1, count_native_2q=0, estimated_error=None)
    w = {"depth": 2.0, "count_1q": 0.5, "count_2q": 3.0, "count_native_2q": 1.0, "error": 50.0}
    expected = 2.0 * 2 + 0.5 * 3 + 3.0 * 1 + 1.0 * 0
    assert c.scalarize(w) == pytest.approx(expected)


def test_scalarize_does_not_mutate_weights():
    w = {"depth": 1.0, "count_1q": 0.1, "count_2q": 1.0, "count_native_2q": 0.5, "error": 100.0}
    original = dict(w)
    Cost(1, 1, 1, 1, None).scalarize(w)
    assert w == original


# ---------------------------------------------------------------------------
# Calibration-driven estimated_error
# ---------------------------------------------------------------------------

def test_estimated_error_two_ops():
    """estimated_error = (1-0.01)*(1-0.05)."""
    class DummyCfg:
        calibration = {"per_gate_error": {"x": 0.01, "cx": 0.05}}

    x_op  = instr("x",  target_qubits=[0])
    cx_op = instr("cx", target_qubits=[1], control_qubits=[0], is_controlled=True)
    c = Cost.measure([x_op, cx_op], "arch", qpu_config=DummyCfg())

    expected = (1.0 - 0.01) * (1.0 - 0.05)
    assert c.estimated_error == pytest.approx(expected, abs=1e-12)


def test_estimated_error_no_calibration(prog4):
    c = Cost.measure(prog4, "arch", qpu_config=None)
    assert c.estimated_error is None


def test_estimated_error_unknown_symbol_treated_as_zero():
    class DummyCfg:
        calibration = {"per_gate_error": {"cx": 0.05}}

    h_op  = instr("h",  target_qubits=[0])   # not in calibration → 0.0 error
    cx_op = instr("cx", target_qubits=[1], control_qubits=[0], is_controlled=True)
    c = Cost.measure([h_op, cx_op], "arch", qpu_config=DummyCfg())
    expected = 1.0 * (1.0 - 0.05)
    assert c.estimated_error == pytest.approx(expected, abs=1e-12)


def test_scalarize_with_error_term():
    from lccfq_lang.opt.cost import DEFAULT_WEIGHTS
    c = Cost(depth=1, count_1q=1, count_2q=0, count_native_2q=0, estimated_error=0.99)
    score = c.scalarize()
    error_term = DEFAULT_WEIGHTS["error"] * (1.0 - 0.99)
    assert score == pytest.approx(
        DEFAULT_WEIGHTS["depth"] * 1
        + DEFAULT_WEIGHTS["count_1q"] * 1
        + error_term
    )


# ---------------------------------------------------------------------------
# measure_counts (Perf #1)
# ---------------------------------------------------------------------------

def test_measure_counts_returns_none_depth():
    """measure_counts on an empty program: depth is None, not 0."""
    c = Cost.measure_counts([], "arch")
    assert c.depth is None
    assert c == Cost(None, 0, 0, 0, None)


def test_measure_counts_counts_match_measure(prog4):
    """measure_counts produces the same count/error fields as measure."""
    full  = Cost.measure(prog4, "arch")
    cheap = Cost.measure_counts(prog4, "arch")
    assert cheap.count_1q        == full.count_1q
    assert cheap.count_2q        == full.count_2q
    assert cheap.count_native_2q == full.count_native_2q
    assert cheap.estimated_error == full.estimated_error
    assert cheap.depth is None
    assert full.depth is not None and full.depth >= 0


def test_measure_counts_does_not_build_dag(monkeypatch, prog4):
    """Belt-and-braces: assert circuit_to_dag is not invoked by the cheap path."""
    import lccfq_lang.opt.cost as cost_mod
    calls = {"n": 0}
    orig = cost_mod.circuit_to_dag
    def spy(p):
        calls["n"] += 1
        return orig(p)
    monkeypatch.setattr(cost_mod, "circuit_to_dag", spy)
    Cost.measure_counts(prog4, "arch")
    assert calls["n"] == 0
    Cost.measure(prog4, "arch")
    assert calls["n"] == 1


def test_measure_counts_with_calibration_returns_error():
    """measure_counts propagates calibration-based estimated_error."""
    class DummyCfg:
        calibration = {"per_gate_error": {"x": 0.01}}
    op = Instruction(symbol="x", target_qubits=[0])
    c = Cost.measure_counts([op], "arch", qpu_config=DummyCfg())
    assert c.estimated_error == pytest.approx(1.0 - 0.01, abs=1e-12)
    assert c.depth is None


def test_measure_counts_empty_with_calibration():
    """measure_counts on empty program + calibration: estimated_error=1.0, depth=None."""
    class DummyCfg:
        calibration = {"per_gate_error": {"x": 0.01}}
    c = Cost.measure_counts([], "arch", qpu_config=DummyCfg())
    assert c == Cost(None, 0, 0, 0, 1.0)
    assert c.depth is None


def test_scalarize_handles_none_depth():
    """scalarize() omits the depth term when depth is None (same idiom as error)."""
    from lccfq_lang.opt.cost import DEFAULT_WEIGHTS
    c = Cost(depth=None, count_1q=2, count_2q=1, count_native_2q=0, estimated_error=None)
    expected = (
        DEFAULT_WEIGHTS["count_1q"] * 2
        + DEFAULT_WEIGHTS["count_2q"] * 1
    )
    assert c.scalarize() == pytest.approx(expected)


def test_scalarize_none_depth_with_error():
    """scalarize() with depth=None and a real error term."""
    from lccfq_lang.opt.cost import DEFAULT_WEIGHTS
    c = Cost(depth=None, count_1q=0, count_2q=0, count_native_2q=0, estimated_error=0.9)
    expected = DEFAULT_WEIGHTS["error"] * (1.0 - 0.9)
    assert c.scalarize() == pytest.approx(expected)


def test_scalarize_none_depth_matches_zero_depth_counts():
    """For pure count-based comparison, None-depth and zero-depth agree on counts."""
    from lccfq_lang.opt.cost import DEFAULT_WEIGHTS
    c_none = Cost(depth=None, count_1q=3, count_2q=2, count_native_2q=1, estimated_error=None)
    c_zero = Cost(depth=0,    count_1q=3, count_2q=2, count_native_2q=1, estimated_error=None)
    # None-depth omits depth term; depth=0 contributes 0 to the sum — both equal.
    assert c_none.scalarize() == pytest.approx(c_zero.scalarize())
