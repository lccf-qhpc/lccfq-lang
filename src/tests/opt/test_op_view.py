"""
Filename: test_op_view.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for OpView — the uniform operation-level view used by the
    optimization infrastructure.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.arch.instruction import Instruction, InstructionType
from lccfq_lang.mach.ir import Gate, Control
from lccfq_lang.mach.ir import Test as MachTest
from lccfq_lang.opt import OpView


# ---------------------------------------------------------------------------
# arch.Instruction kinds
# ---------------------------------------------------------------------------

class TestOpViewArch1Q:
    def setup_method(self):
        self.instr = Instruction(symbol="x", target_qubits=[0])
        self.view = OpView(self.instr)

    def test_kind(self):
        assert self.view.kind == "arch"

    def test_symbol(self):
        assert self.view.symbol == "x"

    def test_targets(self):
        assert self.view.targets == (0,)

    def test_controls(self):
        assert self.view.controls == ()

    def test_qubits(self):
        assert self.view.qubits == (0,)

    def test_is_two_qubit(self):
        assert self.view.is_two_qubit is False

    def test_is_measurement(self):
        assert self.view.is_measurement is False

    def test_is_classical(self):
        assert self.view.is_classical is False


class TestOpViewArch2Q:
    def setup_method(self):
        self.instr = Instruction(
            symbol="cx",
            target_qubits=[1],
            control_qubits=[0],
            is_controlled=True,
        )
        self.view = OpView(self.instr)

    def test_kind(self):
        assert self.view.kind == "arch"

    def test_symbol(self):
        assert self.view.symbol == "cx"

    def test_targets(self):
        assert self.view.targets == (1,)

    def test_controls(self):
        assert self.view.controls == (0,)

    def test_qubits_ordering(self):
        # qubits = controls + targets
        assert self.view.qubits == (0, 1)

    def test_is_two_qubit(self):
        assert self.view.is_two_qubit is True

    def test_is_measurement(self):
        assert self.view.is_measurement is False

    def test_is_classical(self):
        assert self.view.is_classical is False


class TestOpViewArchMeasure:
    def setup_method(self):
        self.instr = Instruction(symbol="measure", target_qubits=[0])
        self.view = OpView(self.instr)

    def test_is_measurement(self):
        assert self.view.is_measurement is True

    def test_is_classical(self):
        assert self.view.is_classical is False

    def test_kind(self):
        assert self.view.kind == "arch"


class TestOpViewArchNoQubits:
    """Instruction with no qubit lists (e.g. QPUSTATE-like)."""
    def setup_method(self):
        self.instr = Instruction(symbol="qpustate", target_qubits=None, control_qubits=None)
        self.view = OpView(self.instr)

    def test_targets_empty(self):
        assert self.view.targets == ()

    def test_controls_empty(self):
        assert self.view.controls == ()

    def test_qubits_empty(self):
        assert self.view.qubits == ()

    def test_is_two_qubit(self):
        assert self.view.is_two_qubit is False

    def test_is_classical(self):
        assert self.view.is_classical is False


# ---------------------------------------------------------------------------
# mach.ir kinds
# ---------------------------------------------------------------------------

class TestOpViewMachGate1Q:
    def setup_method(self):
        self.gate = Gate(symbol="rx", target_qubits=[0], control_qubits=[], params=[0.5])
        self.view = OpView(self.gate)

    def test_kind(self):
        assert self.view.kind == "mach.gate"

    def test_symbol(self):
        assert self.view.symbol == "rx"

    def test_targets(self):
        assert self.view.targets == (0,)

    def test_controls(self):
        assert self.view.controls == ()

    def test_qubits(self):
        assert self.view.qubits == (0,)

    def test_params(self):
        assert self.view.params == (0.5,)

    def test_is_two_qubit(self):
        assert self.view.is_two_qubit is False

    def test_is_measurement(self):
        assert self.view.is_measurement is False

    def test_is_classical(self):
        assert self.view.is_classical is False


class TestOpViewMachGate2Q:
    def setup_method(self):
        self.gate = Gate(
            symbol="sqiswap",
            target_qubits=[1],
            control_qubits=[0],
            params=[],
        )
        self.view = OpView(self.gate)

    def test_kind(self):
        assert self.view.kind == "mach.gate"

    def test_targets(self):
        assert self.view.targets == (1,)

    def test_controls(self):
        assert self.view.controls == (0,)

    def test_qubits_ordering(self):
        # qubits = controls + targets
        assert self.view.qubits == (0, 1)

    def test_is_two_qubit(self):
        assert self.view.is_two_qubit is True

    def test_is_classical(self):
        assert self.view.is_classical is False


class TestOpViewMachControl:
    def setup_method(self):
        self.ctrl = Control(symbol="cond", params=[1])
        self.view = OpView(self.ctrl)

    def test_kind(self):
        assert self.view.kind == "mach.control"

    def test_symbol(self):
        assert self.view.symbol == "cond"

    def test_targets(self):
        assert self.view.targets == ()

    def test_controls(self):
        assert self.view.controls == ()

    def test_qubits(self):
        assert self.view.qubits == ()

    def test_is_classical(self):
        assert self.view.is_classical is True

    def test_is_measurement(self):
        assert self.view.is_measurement is False


class TestOpViewMachTest:
    def setup_method(self):
        self.test_cmd = MachTest(symbol="meas", params=[0], shots=1024)
        self.view = OpView(self.test_cmd)

    def test_kind(self):
        assert self.view.kind == "mach.test"

    def test_symbol(self):
        assert self.view.symbol == "meas"

    def test_targets(self):
        assert self.view.targets == ()

    def test_controls(self):
        assert self.view.controls == ()

    def test_qubits(self):
        assert self.view.qubits == ()

    def test_is_classical(self):
        assert self.view.is_classical is True

    def test_is_measurement(self):
        assert self.view.is_measurement is False


# ---------------------------------------------------------------------------
# Unsupported type
# ---------------------------------------------------------------------------

def test_unsupported_type_raises():
    with pytest.raises(TypeError, match="unsupported op type"):
        OpView(42)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr():
    view = OpView(Instruction(symbol="h", target_qubits=[2]))
    r = repr(view)
    assert "arch" in r
    assert "h" in r
