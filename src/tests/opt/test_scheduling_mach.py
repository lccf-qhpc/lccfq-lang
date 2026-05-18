"""
Filename: test_scheduling_mach.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for DeferMeasurement and ParallelizeLayers scheduling passes.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from lccfq_lang.mach.ir import Gate, Control
from lccfq_lang.opt.builtin.scheduling_mach import (
    DeferMeasurement,
    ParallelizeLayers,
)
from lccfq_lang.opt.pass_base import PassContext


class _StubISA:
    pass


def _ctx():
    return PassContext()


# ---- DeferMeasurement ----

def test_defer_measurement_moves_to_end():
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("measure", [0], None, []),
        Gate("ry", [1], None, [0.2]),
    ]
    out, _ = DeferMeasurement(_StubISA()).run(p, _ctx())
    assert [op.symbol for op in out] == ["rx", "ry", "measure"]


def test_defer_measurement_preserves_measure_order():
    p = [
        Gate("measure", [0], None, []),
        Gate("rx", [0], None, [0.1]),
        Gate("measure", [1], None, []),
    ]
    out, _ = DeferMeasurement(_StubISA()).run(p, _ctx())
    assert [op.symbol for op in out] == ["rx", "measure", "measure"]
    assert out[1].target_qubits == [0]
    assert out[2].target_qubits == [1]


def test_defer_measurement_no_measures_passes_through():
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("ry", [1], None, [0.2]),
    ]
    out, _ = DeferMeasurement(_StubISA()).run(p, _ctx())
    assert [op.symbol for op in out] == ["rx", "ry"]


def test_defer_measurement_only_measures():
    p = [
        Gate("measure", [0], None, []),
        Gate("measure", [1], None, []),
    ]
    out, _ = DeferMeasurement(_StubISA()).run(p, _ctx())
    assert [op.symbol for op in out] == ["measure", "measure"]


def test_defer_measurement_empty_program():
    out, _ = DeferMeasurement(_StubISA()).run([], _ctx())
    assert out == []


def test_defer_measurement_does_not_move_reset():
    """reset is a state-prep primitive; it must NOT be moved."""
    p = [
        Gate("measure", [0], None, []),
        Gate("reset", [0], None, []),
        Gate("rx", [0], None, [0.1]),
    ]
    out, _ = DeferMeasurement(_StubISA()).run(p, _ctx())
    # measure moves to end; reset and rx stay in original relative order.
    symbols = [op.symbol for op in out]
    assert symbols == ["reset", "rx", "measure"]


def test_defer_measurement_classical_ops_stay():
    """Control ops must not be moved."""
    ctrl = Control("shot", [100])
    p = [
        Gate("measure", [0], None, []),
        ctrl,
        Gate("rx", [0], None, [0.5]),
    ]
    out, _ = DeferMeasurement(_StubISA()).run(p, _ctx())
    # ctrl and rx are non-measure, measure goes to end.
    assert out[0] is ctrl
    assert out[1].symbol == "rx"
    assert out[2].symbol == "measure"


def test_defer_measurement_returns_new_list():
    """The returned list must be a different object (no aliasing)."""
    p = [Gate("rx", [0], None, [0.1])]
    out, _ = DeferMeasurement(_StubISA()).run(p, _ctx())
    assert out is not p


# ---- ParallelizeLayers ----

def test_parallelize_layers_writes_tags():
    p = [
        Gate("rx", [0], None, [0.1]),       # layer 0
        Gate("ry", [1], None, [0.2]),       # layer 0 (independent)
        Gate("sqiswap", [0], [1], []),      # layer 1
        Gate("rx", [0], None, [0.3]),       # layer 2
    ]
    out, _ = ParallelizeLayers(_StubISA()).run(p, _ctx())
    assert out[0].tags["layer"] == 0
    assert out[1].tags["layer"] == 0
    assert out[2].tags["layer"] == 1
    assert out[3].tags["layer"] == 2


def test_parallelize_layers_does_not_reorder():
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("ry", [1], None, [0.2]),
        Gate("rx", [2], None, [0.3]),
    ]
    out, _ = ParallelizeLayers(_StubISA()).run(p, _ctx())
    # Same identity preservation: tags written in place.
    for orig, new in zip(p, out):
        assert orig is new


def test_parallelize_layers_single_qubit_chain():
    """Sequential ops on the same qubit get ascending layer numbers."""
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("ry", [0], None, [0.2]),
        Gate("rx", [0], None, [0.3]),
    ]
    out, _ = ParallelizeLayers(_StubISA()).run(p, _ctx())
    layers = [op.tags["layer"] for op in out]
    assert layers == [0, 1, 2]


def test_parallelize_layers_empty_program():
    out, _ = ParallelizeLayers(_StubISA()).run([], _ctx())
    assert out == []


def test_parallelize_layers_returns_new_list():
    """The returned list must be a different list object."""
    p = [Gate("rx", [0], None, [0.1])]
    out, _ = ParallelizeLayers(_StubISA()).run(p, _ctx())
    assert out is not p


def test_parallelize_layers_all_parallel():
    """All ops on distinct qubits get layer 0."""
    p = [
        Gate("rx", [0], None, [0.1]),
        Gate("rx", [1], None, [0.2]),
        Gate("rx", [2], None, [0.3]),
    ]
    out, _ = ParallelizeLayers(_StubISA()).run(p, _ctx())
    for op in out:
        assert op.tags["layer"] == 0


def test_parallelize_layers_classical_not_tagged():
    """Control ops have no tags attribute and must not be tagged."""
    ctrl = Control("shot", [100])
    p = [
        Gate("rx", [0], None, [0.1]),
        ctrl,
    ]
    out, _ = ParallelizeLayers(_StubISA()).run(p, _ctx())
    assert out[0].tags["layer"] == 0
    assert not hasattr(ctrl, "tags") or "layer" not in ctrl.tags


def test_parallelize_layers_mutates_in_place():
    """Since ParallelizeLayers mutates Gate.tags in place, the original
    Gate objects should also see the tags after the pass runs."""
    g = Gate("rx", [0], None, [0.1])
    p = [g]
    _, _ = ParallelizeLayers(_StubISA()).run(p, _ctx())
    assert g.tags.get("layer") == 0
