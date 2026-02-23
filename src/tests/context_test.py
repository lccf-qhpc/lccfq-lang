"""
Filename: context_test.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    Tests for CompilationPipeline, CompilerPass, Circuit and Test contexts.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from pathlib import Path

from lccfq_lang.arch.context import CompilerPass, CompilationPipeline, Circuit, Test
from lccfq_lang.arch.register import CRegister, QContext
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.error import UnknownCompilerPass
from lccfq_lang.backend import QPU


DATA_DIR = Path(__file__).parent / "data"
CONFIG = str(DATA_DIR / "testing.toml")


# ──────────────────────────────────────────────
# CompilationPipeline unit tests (no QPU needed)
# ──────────────────────────────────────────────

def _upper_pass(items):
    return [s.upper() for s in items]


def _double_pass(items):
    return items + items


def test_pipeline_single_pass():
    pipeline = CompilationPipeline([
        CompilerPass("step1", _upper_pass),
    ])
    name, result = pipeline.run(["a", "b"], "step1")
    assert name == "step1"
    assert result == ["A", "B"]


def test_pipeline_stops_at_requested_pass():
    pipeline = CompilationPipeline([
        CompilerPass("step1", _upper_pass),
        CompilerPass("step2", _double_pass),
    ])
    name, result = pipeline.run(["a"], "step1")
    assert name == "step1"
    assert result == ["A"]


def test_pipeline_runs_through_multiple_passes():
    pipeline = CompilationPipeline([
        CompilerPass("step1", _upper_pass),
        CompilerPass("step2", _double_pass),
    ])
    name, result = pipeline.run(["a"], "step2")
    assert name == "step2"
    assert result == ["A", "A"]


def test_pipeline_unknown_pass_raises():
    pipeline = CompilationPipeline([
        CompilerPass("step1", _upper_pass),
    ])
    with pytest.raises(UnknownCompilerPass):
        pipeline.run(["a"], "nonexistent")


def test_pipeline_identity_pass():
    pipeline = CompilationPipeline([
        CompilerPass("noop", lambda prog: prog),
    ])
    data = [1, 2, 3]
    name, result = pipeline.run(data, "noop")
    assert result == [1, 2, 3]


# ──────────────────────────────────────────────
# Circuit context integration tests
# ──────────────────────────────────────────────

@pytest.fixture
def qpu_parsed():
    return QPU(filename=CONFIG, last_pass="parsed")


@pytest.fixture
def qpu_mapped():
    return QPU(filename=CONFIG, last_pass="mapped")


@pytest.fixture
def qpu_transpiled():
    return QPU(filename=CONFIG, last_pass="transpiled")


def test_circuit_parsed_pass_populates_creg(qpu_parsed):
    qpu = qpu_parsed
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu, shots=100) as c:
        c >> qpu.isa.x(tg=0)

    # parsed pass produces -1 sentinel values
    assert creg.data is not None
    assert all(v == -1 for v in creg.data.values())


def test_circuit_mapped_pass_populates_creg(qpu_mapped):
    qpu = qpu_mapped
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu, shots=100) as c:
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.cx(ct=0, tg=1)

    assert creg.data is not None
    assert all(v == -1 for v in creg.data.values())


def test_circuit_transpiled_pass_populates_creg(qpu_transpiled):
    qpu = qpu_transpiled
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu, shots=100) as c:
        c >> qpu.isa.x(tg=0)
        c >> qpu.isa.measure(tgs=[0, 1])

    assert creg.data is not None
    assert all(v == -1 for v in creg.data.values())


def test_circuit_unknown_pass_raises():
    qpu = QPU(filename=CONFIG, last_pass="bogus_pass")
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with pytest.raises(UnknownCompilerPass):
        with Circuit(qreg, creg, qpu) as c:
            c >> qpu.isa.x(tg=0)


def test_circuit_propagates_exception_from_body(qpu_transpiled):
    """Exceptions raised inside the with-block must not be suppressed."""
    qpu = qpu_transpiled
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with pytest.raises(ValueError, match="test error"):
        with Circuit(qreg, creg, qpu) as c:
            raise ValueError("test error")


def test_circuit_instructions_are_challenged(qpu_transpiled):
    """Instructions appended via >> should be challenged (deep-copied with correct type)."""
    qpu = qpu_transpiled
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    original = qpu.isa.x(tg=0)

    with Circuit(qreg, creg, qpu) as c:
        c >> original

    # The stored instruction should be a different object (deep copy)
    assert c.instructions[0] is not original
    # And should have CIRCUIT instruction type
    from lccfq_lang.arch.instruction import InstructionType
    assert c.instructions[0].instruction_type == InstructionType.CIRCUIT


def test_circuit_results_and_frequencies(qpu_parsed):
    qpu = qpu_parsed
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu) as c:
        c >> qpu.isa.x(tg=0)

    assert c.results() is not None
    # frequencies() should not raise on sentinel data
    freqs = c.frequencies()
    assert isinstance(freqs, dict)


# ──────────────────────────────────────────────
# Test context integration tests
# ──────────────────────────────────────────────

def test_test_context_populates_accumulator(qpu_transpiled):
    qpu = qpu_transpiled
    qreg = qpu.qregister(2)
    accum = {}

    with Test(qreg, accum, qpu) as t:
        t >> qpu.isa.x(tg=0, shots=100)

    # exec_single is a no-op returning None, but the accumulator should have an entry
    assert 0 in accum


def test_test_context_multiple_instructions(qpu_transpiled):
    qpu = qpu_transpiled
    qreg = qpu.qregister(2)
    accum = {}

    with Test(qreg, accum, qpu) as t:
        t >> qpu.isa.x(tg=0, shots=100)
        t >> qpu.isa.h(tg=1, shots=200)

    assert len(accum) == 2
    assert 0 in accum
    assert 1 in accum
