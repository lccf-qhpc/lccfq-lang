"""
Filename: test_native_synthesis.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for the RyRzRyToHardware native synthesis pass.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import math
from lccfq_lang.mach.ir import Gate
from lccfq_lang.opt.builtin.native_synthesis import RyRzRyToHardware
from lccfq_lang.opt.pass_base import PassContext
from tests.opt._equiv_native import assert_equivalent_native


class _StubISA:
    pass


def _ctx():
    return PassContext()


def _band(q, alpha):
    """The 3-gate band that XYiSW emits for rz(alpha)."""
    return [
        Gate("ry", [q], None, [-math.pi / 2]),
        Gate("rx", [q], None, [alpha]),
        Gate("ry", [q], None, [+math.pi / 2]),
    ]


def test_ry_rz_ry_collapses_two_bands():
    p = _band(0, 0.7) + _band(0, 0.4)
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    assert len(out) == 3
    assert out[0].symbol == "ry" and math.isclose(out[0].params[0], -math.pi / 2)
    assert out[1].symbol == "rx" and math.isclose(out[1].params[0], 1.1, abs_tol=1e-9)
    assert out[2].symbol == "ry" and math.isclose(out[2].params[0], +math.pi / 2)


def test_ry_rz_ry_does_not_fire_on_different_qubits():
    p = _band(0, 0.7) + _band(1, 0.4)
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    assert len(out) == 6


def test_ry_rz_ry_semantic_equivalence():
    p = _band(0, 0.7) + _band(0, 0.4)
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    assert_equivalent_native(p, out, n_qubits=1)


def test_ry_rz_ry_does_not_fire_on_single_band():
    p = _band(0, 0.7)
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    assert len(out) == 3


def test_ry_rz_ry_handles_tail():
    # A non-pattern op after a single band must remain.
    p = _band(0, 0.7) + [Gate("rx", [0], None, [0.1])]
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    assert [op.symbol for op in out] == ["ry", "rx", "ry", "rx"]


def test_ry_rz_ry_empty_program():
    out, _ = RyRzRyToHardware(_StubISA()).run([], _ctx())
    assert out == []


def test_ry_rz_ry_three_consecutive_bands():
    """Three consecutive bands should collapse to one band per pass."""
    p = _band(0, 0.3) + _band(0, 0.4) + _band(0, 0.5)
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    # First pass collapses the first two bands; the third stays.
    assert len(out) == 6
    assert_equivalent_native(p, out, n_qubits=1)


def test_ry_rz_ry_correct_angle_sum_modulo():
    """Result rx angle must be MOD_2PI(alpha+beta)."""
    alpha = 1.5
    beta = 2.0
    p = _band(0, alpha) + _band(0, beta)
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    assert len(out) == 3
    expected = (alpha + beta) % (2 * math.pi)
    if expected > math.pi:
        expected -= 2 * math.pi
    actual = out[1].params[0]
    assert math.isclose(actual, expected, abs_tol=1e-9)
    assert_equivalent_native(p, out, n_qubits=1)


def test_ry_rz_ry_pattern_requires_correct_ry_signs():
    """Modifying one ry sign breaks the pattern — no collapse."""
    p = [
        Gate("ry", [0], None, [+math.pi / 2]),  # wrong sign for first ry
        Gate("rx", [0], None, [0.7]),
        Gate("ry", [0], None, [+math.pi / 2]),
        Gate("ry", [0], None, [-math.pi / 2]),
        Gate("rx", [0], None, [0.4]),
        Gate("ry", [0], None, [+math.pi / 2]),
    ]
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    assert len(out) == 6  # no collapse


def test_ry_rz_ry_non_gate_in_window_prevents_collapse():
    """A non-Gate op inside the 6-gate window must prevent the pattern."""
    from lccfq_lang.mach.ir import Control
    ctrl = Control("shot", [100])
    p = [
        Gate("ry", [0], None, [-math.pi / 2]),
        Gate("rx", [0], None, [0.7]),
        ctrl,  # not a Gate — breaks the window
        Gate("ry", [0], None, [-math.pi / 2]),
        Gate("rx", [0], None, [0.4]),
        Gate("ry", [0], None, [+math.pi / 2]),
    ]
    out, _ = RyRzRyToHardware(_StubISA()).run(p, _ctx())
    # No collapse; should have 6 ops (ctrl counts)
    assert len(out) == 6
