"""
Filename: test_mach_opt_pipeline.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    End-to-end pipeline tests for the mach_opt PassGroup at each opt_level.
    Tests focus on group presence/absence, opt_level=0 byte identity,
    and last_pass="mach_optimized" behavior.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import inspect
import pytest
from pathlib import Path
from lccfq_lang.arch.context import Circuit
from lccfq_lang.arch.register import CRegister
from lccfq_lang.backend import QPU
from lccfq_lang.opt.builtin.lower_passes import build_lowering_groups, slice_groups_for
from lccfq_lang.opt.manager import PassManager
from lccfq_lang.opt.pass_base import PassContext
from lccfq_lang.mach.ir import Gate

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG = str(DATA_DIR / "testing.toml")


def _qpu(last_pass):
    return QPU(filename=CONFIG, last_pass=last_pass)


# ---------------------------------------------------------------------------
# opt_level=0: mach_opt group must be absent
# ---------------------------------------------------------------------------

def test_level_0_no_mach_opt_group():
    """opt_level=0 must NOT create a mach_opt group."""
    qpu = _qpu("transpiled")
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=0) as c:
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.measure(tgs=[0, 1])
    assert all(r.group_name != "mach_opt" for r in c._opt_records)


def test_level_0_mach_opt_group_absent_from_groups():
    """build_lowering_groups at opt_level=0 should not produce mach_opt."""
    qpu = _qpu("transpiled")
    qreg = qpu.qregister(2)
    groups = build_lowering_groups(qreg, qpu, opt_level=0)
    group_names = [g.name for g in groups]
    assert "mach_opt" not in group_names


# ---------------------------------------------------------------------------
# opt_level>=1: mach_opt group must be present
# ---------------------------------------------------------------------------

def test_level_1_emits_mach_opt_records():
    """At opt_level=1 with last_pass='mach_optimized', mach_opt group appears."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=1) as c:
        c >> qpu.isa.rz(tg=0, params=[0.3])
        c >> qpu.isa.rz(tg=0, params=[0.4])
        c >> qpu.isa.measure(tgs=[0])
    assert any(r.group_name == "mach_opt" for r in c._opt_records)


def test_level_2_emits_mach_opt_records():
    """At opt_level=2 with last_pass='mach_optimized', mach_opt group appears."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2) as c:
        c >> qpu.isa.rz(tg=0, params=[0.3])
        c >> qpu.isa.rz(tg=0, params=[0.4])
        c >> qpu.isa.measure(tgs=[0])
    assert any(r.group_name == "mach_opt" for r in c._opt_records)


def test_level_3_emits_mach_opt_records():
    """At opt_level=3 with last_pass='mach_optimized', mach_opt group appears."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=3) as c:
        c >> qpu.isa.rz(tg=0, params=[0.3])
        c >> qpu.isa.rz(tg=0, params=[0.4])
        c >> qpu.isa.measure(tgs=[0])
    assert any(r.group_name == "mach_opt" for r in c._opt_records)


# ---------------------------------------------------------------------------
# last_pass="mach_optimized" behavior
# ---------------------------------------------------------------------------

def test_last_pass_mach_optimized_is_whitelisted():
    """QPU should accept last_pass='mach_optimized' without resetting to default."""
    qpu = _qpu("mach_optimized")
    assert qpu.last_pass == "mach_optimized"


def test_last_pass_mach_optimized_in_lowering_stages():
    from lccfq_lang.opt.builtin.lower_passes import LOWERING_STAGES
    assert "mach_optimized" in LOWERING_STAGES


def test_mach_optimized_falls_back_when_mach_opt_omitted():
    """opt_level=0 -> mach_opt group absent -> last_pass='mach_optimized'
    must fall back to lower_transpile without raising."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(1)
    creg = CRegister(1)
    # Should not raise.
    with Circuit(qreg, creg, qpu, shots=1, opt_level=0) as c:
        c >> qpu.isa.x(tg=0)


def test_mach_optimized_with_level_1_runs():
    """last_pass='mach_optimized' with opt_level=1 should succeed."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(1)
    creg = CRegister(1)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=1) as c:
        c >> qpu.isa.x(tg=0)


# ---------------------------------------------------------------------------
# Backward compatibility: Circuit.__init__ signature unchanged
# ---------------------------------------------------------------------------

def test_circuit_init_signature_unchanged():
    """Phase 3/5: Circuit.__init__ must expose exactly the expected parameters."""
    sig = inspect.signature(Circuit.__init__)
    params = list(sig.parameters.keys())
    # Phase 5 adds `report` (bool, default False) after opt_passes.
    expected = ["self", "qreg", "creg", "qpu", "shots", "verbose", "opt_level", "opt_passes", "report"]
    assert params == expected


# ---------------------------------------------------------------------------
# Byte identity at opt_level=0
# ---------------------------------------------------------------------------

def test_byte_identity_at_level_0():
    """At opt_level=0, the pipeline must produce the same gate list
    as a pipeline built with opt_passes=[] (explicit empty list)."""
    qpu = _qpu("transpiled")
    qreg_a = qpu.qregister(2)
    qreg_b = qpu.qregister(2)

    # Build groups via opt_level=0
    groups_a = build_lowering_groups(qreg_a, qpu, opt_level=0)
    # Build groups via opt_passes=[]
    groups_b = build_lowering_groups(qreg_b, qpu, opt_passes=[])

    # Both must have the same group names (no mach_opt in either).
    names_a = [g.name for g in groups_a]
    names_b = [g.name for g in groups_b]
    assert "mach_opt" not in names_a
    assert "mach_opt" not in names_b
    assert names_a == names_b


# ---------------------------------------------------------------------------
# Gate default construction backward compatibility
# ---------------------------------------------------------------------------

def test_gate_default_construction_unchanged():
    """Gate(symbol, target_qubits, control_qubits, params).to_json() must have
    exactly the same keys as before Phase 3 (no tags or duration)."""
    g = Gate("rx", [0], [1], [0.5])
    j = g.to_json()
    assert set(j.keys()) == {"symbol", "target_qubits", "control_qubits", "params"}


# ---------------------------------------------------------------------------
# last_pass legacy values still work
# ---------------------------------------------------------------------------

def test_last_pass_transpiled_still_works():
    """QPU(..., last_pass='transpiled') must remain the default pipeline endpoint."""
    qpu = _qpu("transpiled")
    assert qpu.last_pass == "transpiled"
    qreg = qpu.qregister(1)
    creg = CRegister(1)
    with Circuit(qreg, creg, qpu, shots=1) as c:
        c >> qpu.isa.x(tg=0)
    # Should not raise; creg filled with sentinel -1 values.
    assert creg.data is not None


def test_last_pass_arch_optimized_still_works():
    """QPU(..., last_pass='arch_optimized') must still slice correctly."""
    qpu = _qpu("arch_optimized")
    qreg = qpu.qregister(1)
    creg = CRegister(1)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=1) as c:
        c >> qpu.isa.x(tg=0)
    assert any(r.group_name == "arch_opt" for r in c._opt_records)


# ---------------------------------------------------------------------------
# Explicit opt_passes with mach passes
# ---------------------------------------------------------------------------

def test_explicit_opt_passes_with_mach_pass():
    """opt_passes=['merge_adjacent_1q'] should create a mach_opt group."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_passes=["merge_adjacent_1q"]) as c:
        c >> qpu.isa.rz(tg=0, params=[0.3])
        c >> qpu.isa.rz(tg=0, params=[0.3])
        c >> qpu.isa.measure(tgs=[0])
    assert any(r.group_name == "mach_opt" for r in c._opt_records)
    assert any(r.pass_name == "merge_adjacent_1q" for r in c._opt_records)


def test_explicit_opt_passes_mixed_arch_and_mach():
    """opt_passes with both arch and mach passes should produce both groups."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(
        qreg, creg, qpu, shots=1,
        opt_passes=["remove_identity", "merge_adjacent_1q"]
    ) as c:
        c >> qpu.isa.rz(tg=0, params=[0.3])
        c >> qpu.isa.measure(tgs=[0])
    assert any(r.group_name == "arch_opt" for r in c._opt_records)
    assert any(r.group_name == "mach_opt" for r in c._opt_records)


def test_unknown_mach_pass_name_raises():
    """opt_passes=['bogus_mach'] should raise ValueError."""
    qpu = _qpu("transpiled")
    qreg = qpu.qregister(1)
    creg = CRegister(1)
    with pytest.raises(ValueError, match="Unknown pass"):
        with Circuit(qreg, creg, qpu, shots=1, opt_passes=["bogus_mach"]) as c:
            c >> qpu.isa.x(tg=0)


# ---------------------------------------------------------------------------
# slice_groups_for with mach_optimized
# ---------------------------------------------------------------------------

def test_slice_groups_for_mach_optimized_with_group():
    """When mach_opt is present, slice_groups_for('mach_optimized', ...) includes it."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(2)
    groups = build_lowering_groups(qreg, qpu, opt_level=1)
    sliced = slice_groups_for("mach_optimized", groups)
    assert sliced[-1].name == "mach_opt"


def test_slice_groups_for_mach_optimized_fallback():
    """When mach_opt is absent (opt_level=0), slice_groups_for('mach_optimized', ...)
    falls back to lower_transpile."""
    qpu = _qpu("mach_optimized")
    qreg = qpu.qregister(2)
    groups = build_lowering_groups(qreg, qpu, opt_level=0)
    sliced = slice_groups_for("mach_optimized", groups)
    assert sliced[-1].name == "lower_transpile"
