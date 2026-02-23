"""
Filename: synthesis_test.py
Author: Santiago Nunez-Corrales
Date: 2025-08-06
Version: 1.0
Description:
    Test for OpenQASM 3.0 generation.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.arch.synth.qasm import QASMSynthesizer
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.context import Circuit
from lccfq_lang.arch.register import QRegister, CRegister
from lccfq_lang.arch.error import UnknownInstruction
from lccfq_lang.mach.sets.xyisqswap import XYiSW
from lccfq_lang.backend import QPU


@pytest.fixture
def synth():
    return QASMSynthesizer()


def test_qasm_header(synth):
    header = synth.get_qasm_header(3, 3)
    assert "OPENQASM 3.0;" in header[0]
    assert "qubit[3] q;" in header[1]
    assert "bit[3] c;" in header[2]


def test_sqg(synth):
    instr = Instruction(symbol="x", target_qubits=[0])
    qasm = synth.synth_instruction(instr)
    assert qasm.strip() == "x q[0];"


def test_sqg_par(synth):
    instr = Instruction(symbol="rx", target_qubits=[0], parameters=[1.57])
    qasm = synth.synth_instruction(instr)
    assert qasm.strip() == "rx(1.57) q[0];"


def test_tqg(synth):
    instr = Instruction(symbol="cx", control_qubits=[0], target_qubits=[1])
    qasm = synth.synth_instruction(instr)
    assert qasm.strip() == "cx q[0] , q[1];"


def test_measurement(synth):
    instr = Instruction(symbol="measure", target_qubits=[0, 1])
    qasm = synth.synth_instruction(instr)
    expected = "measure q[0] -> c[0];\nmeasure q[1] -> c[1];"
    assert qasm.strip() == expected


def test_reset(synth):
    instr = Instruction(symbol="reset", target_qubits=[0, 2])
    qasm = synth.synth_instruction(instr)
    expected = "reset q[0];\nreset q[2];"
    assert qasm.strip() == expected


def test_unsupported_instruction(synth):
    instr = Instruction(symbol="foobar", target_qubits=[0])
    with pytest.raises(UnknownInstruction) as excinfo:
        synth.synth_instruction(instr)
    assert "Unrecognized instruction" in str(excinfo.value)


def test_full_circuit_translation(synth):
    qpu = QPU(filename="src/tests/data/testing.toml")
    qreg = QRegister(2, qpu)
    creg = CRegister(2)

    assert qpu.transpiler is not None
    assert isinstance(qpu.transpiler, XYiSW), f"Expected XYiSW transpiler, got {type(qpu.transpiler)}"

    with Circuit(qreg, creg) as c:
        c >> Instruction(symbol="x", target_qubits=[0])
        c >> Instruction(symbol="cx", control_qubits=[0], target_qubits=[1])
        c >> Instruction(symbol="measure", target_qubits=[0, 1])

    qasm = synth.synth_circuit(circuit=c, path="./test.qasm")

    assert "OPENQASM 3.0;" in qasm
    assert "qubit[2] q;" in qasm
    assert "bit[2] c;" in qasm
    assert "x q[0];" in qasm
    assert "cx q[0] , q[1];" in qasm
    assert "measure q[0] -> c[0];" in qasm
    assert "measure q[1] -> c[1];" in qasm