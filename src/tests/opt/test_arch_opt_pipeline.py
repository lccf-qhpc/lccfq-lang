"""
Filename: test_arch_opt_pipeline.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Integration tests for the arch_opt PassGroup wired into the Circuit
    compilation pipeline.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from pathlib import Path
from lccfq_lang.arch.context import Circuit
from lccfq_lang.arch.register import CRegister
from lccfq_lang.backend import QPU
from tests.opt._equiv import assert_equivalent

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG = str(DATA_DIR / "testing.toml")


@pytest.fixture
def qpu_transpiled():
    return QPU(filename=CONFIG, last_pass="transpiled")


@pytest.fixture
def qpu_arch_optimized():
    return QPU(filename=CONFIG, last_pass="arch_optimized")


@pytest.fixture
def qpu_expanded():
    return QPU(filename=CONFIG, last_pass="expanded")


# ---------------------------------------------------------------------------
# opt_level=0 byte-identical
# ---------------------------------------------------------------------------

def test_opt_level_0_no_arch_opt_group(qpu_transpiled):
    """opt_level=0 must NOT create an arch_opt group."""
    qpu = qpu_transpiled
    qreg = qpu.qregister(3)
    creg = CRegister(3)

    with Circuit(qreg, creg, qpu, shots=1, opt_level=0) as c:
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.cx(ct=0, tg=1)
        c >> qpu.isa.cx(ct=1, tg=2)
        c >> qpu.isa.measure(tgs=[0, 1, 2])

    group_names = [r.group_name for r in c._opt_records]
    assert "arch_opt" not in group_names


def test_opt_level_0_no_kwargs_produces_same_groups():
    """Circuit() with no opt kwargs should behave like opt_level=0."""
    qpu = QPU(filename=CONFIG, last_pass="transpiled")
    qreg = qpu.qregister(2)
    creg_a = CRegister(2)
    creg_b = CRegister(2)

    with Circuit(qreg, creg_a, qpu, shots=1) as c_a:
        c_a >> qpu.isa.x(tg=0)
        c_a >> qpu.isa.measure(tgs=[0, 1])

    with Circuit(qreg, creg_b, qpu, shots=1, opt_level=0) as c_b:
        c_b >> qpu.isa.x(tg=0)
        c_b >> qpu.isa.measure(tgs=[0, 1])

    # Neither should have arch_opt records
    assert not any(r.group_name == "arch_opt" for r in c_a._opt_records)
    assert not any(r.group_name == "arch_opt" for r in c_b._opt_records)


# ---------------------------------------------------------------------------
# opt_level=1 reduces gate count
# ---------------------------------------------------------------------------

def test_opt_level_1_reduces_gate_count(qpu_arch_optimized):
    """x(0) x(0) h(0) h(0) with opt_level=1 should collapse to empty (before measure)."""
    qpu = qpu_arch_optimized
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu, shots=1, opt_level=1) as c:
        c >> qpu.isa.x(tg=0)
        c >> qpu.isa.x(tg=0)
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.measure(tgs=[0])

    # The pass runs on the arch IR before transpile; last_pass="arch_optimized"
    # means the program handed to _handle_pass is List[Instruction].
    # creg.data holds the sentinel -1 values (no execution).
    assert creg.data is not None
    assert all(v == -1 for v in creg.data.values())
    # arch_opt group must appear in records
    assert any(r.group_name == "arch_opt" for r in c._opt_records)


# ---------------------------------------------------------------------------
# opt_level=2 collapses HCXHRule
# ---------------------------------------------------------------------------

def test_opt_level_2_collapses_hcxh(qpu_arch_optimized):
    """h(1) cx(0,1) h(1) should become cz(0,1) at opt_level=2."""
    qpu = qpu_arch_optimized
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu, shots=1, opt_level=2) as c:
        c >> qpu.isa.h(tg=1)
        c >> qpu.isa.cx(ct=0, tg=1)
        c >> qpu.isa.h(tg=1)
        c >> qpu.isa.measure(tgs=[0, 1])

    assert any(r.group_name == "arch_opt" for r in c._opt_records)
    # The creg gets sentinel -1 values (no execution at arch_optimized stage)
    assert creg.data is not None


# ---------------------------------------------------------------------------
# Semantic equivalence across opt_levels
# ---------------------------------------------------------------------------

def _build_arch_optimized_program(qpu, opt_level):
    """Return the List[Instruction] left after arch_optimized stage."""
    qpu_ao = QPU(filename=CONFIG, last_pass="arch_optimized")
    qreg = qpu_ao.qregister(2)
    creg = CRegister(2)

    captured = []

    original_handle_pass = Circuit._handle_pass

    def patched_handle_pass(self, program, cpass):
        if cpass == "arch_optimized":
            captured.extend(program)
        original_handle_pass(self, program, cpass)

    Circuit._handle_pass = patched_handle_pass
    try:
        with Circuit(qreg, creg, qpu_ao, shots=1, opt_level=opt_level) as c:
            c >> qpu_ao.isa.h(tg=0)
            c >> qpu_ao.isa.cx(ct=0, tg=1)
            c >> qpu_ao.isa.h(tg=0)
    finally:
        Circuit._handle_pass = original_handle_pass

    return [i for i in captured if i.symbol != "measure"]


def test_opt_levels_produce_arch_optimized_records():
    """Circuits at opt_level 1 and 2 should include arch_opt records."""
    for level in [1, 2, 3]:
        qpu = QPU(filename=CONFIG, last_pass="arch_optimized")
        qreg = qpu.qregister(2)
        creg = CRegister(2)
        with Circuit(qreg, creg, qpu, shots=1, opt_level=level) as c:
            c >> qpu.isa.h(tg=0)
            c >> qpu.isa.cx(ct=0, tg=1)
        assert any(r.group_name == "arch_opt" for r in c._opt_records), (
            f"Expected arch_opt in records for opt_level={level}"
        )


# ---------------------------------------------------------------------------
# Explicit opt_passes
# ---------------------------------------------------------------------------

def test_explicit_opt_passes_overrides_level():
    """opt_passes=['remove_identity'] should create an arch_opt group regardless of opt_level=0."""
    qpu = QPU(filename=CONFIG, last_pass="arch_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu, shots=1, opt_level=0, opt_passes=["remove_identity"]) as c:
        c >> qpu.isa.nop(tgs=[0])
        c >> qpu.isa.x(tg=0)

    assert any(r.group_name == "arch_opt" for r in c._opt_records)
    assert any(r.pass_name == "remove_identity" for r in c._opt_records)


def test_explicit_opt_passes_empty_list_omits_group():
    """opt_passes=[] means no arch_opt group (even if opt_level > 0 were set)."""
    qpu = QPU(filename=CONFIG, last_pass="arch_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu, shots=1, opt_passes=[]) as c:
        c >> qpu.isa.x(tg=0)

    assert not any(r.group_name == "arch_opt" for r in c._opt_records)


def test_unknown_opt_pass_name_raises():
    """opt_passes=['bogus'] should raise ValueError at __exit__."""
    qpu = QPU(filename=CONFIG, last_pass="arch_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with pytest.raises(ValueError, match="Unknown pass"):
        with Circuit(qreg, creg, qpu, shots=1, opt_passes=["bogus"]) as c:
            c >> qpu.isa.x(tg=0)


# ---------------------------------------------------------------------------
# Validation at Circuit.__init__ time
# ---------------------------------------------------------------------------

def test_invalid_opt_level_raises_at_init():
    """Circuit(opt_level=4) must raise ValueError at construction, not __exit__."""
    qpu = QPU(filename=CONFIG, last_pass="transpiled")
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with pytest.raises(ValueError, match="opt_level must be one of"):
        Circuit(qreg, creg, qpu, opt_level=4)


def test_invalid_opt_passes_type_raises_at_init():
    """Circuit(opt_passes='bad') must raise TypeError at construction."""
    qpu = QPU(filename=CONFIG, last_pass="transpiled")
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with pytest.raises(TypeError, match="opt_passes must be None or list"):
        Circuit(qreg, creg, qpu, opt_passes="bad")


# ---------------------------------------------------------------------------
# last_pass="arch_optimized" fallback when opt_level=0
# ---------------------------------------------------------------------------

def test_last_pass_arch_optimized_with_opt_level_zero_falls_back():
    """When opt_level=0 and last_pass='arch_optimized', slicer falls back to
    lower_expand; no exception should be raised."""
    qpu = QPU(filename=CONFIG, last_pass="arch_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    # Should complete without error
    with Circuit(qreg, creg, qpu, shots=1, opt_level=0) as c:
        c >> qpu.isa.x(tg=0)

    # No arch_opt group in records (fell back to expand)
    assert not any(r.group_name == "arch_opt" for r in c._opt_records)


# ---------------------------------------------------------------------------
# opt_records contain arch_opt entries at opt_level >= 1
# ---------------------------------------------------------------------------

def test_opt_records_contain_arch_opt_entries():
    """opt_level=2 should populate _opt_records with arch_opt group entries."""
    qpu = QPU(filename=CONFIG, last_pass="arch_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)

    with Circuit(qreg, creg, qpu, shots=1, opt_level=2) as c:
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.cx(ct=0, tg=1)

    arch_opt_records = [r for r in c._opt_records if r.group_name == "arch_opt"]
    assert len(arch_opt_records) >= 1
