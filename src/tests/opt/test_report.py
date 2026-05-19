"""
Filename: test_report.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.0
Description:
    Tests for Phase 5 reporting: Circuit(report=...) kwarg, the structured
    Circuit.opt_report dict, backward compatibility, and edge cases.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import json
import math
import pytest
from pathlib import Path

from lccfq_lang.arch.context import Circuit
from lccfq_lang.arch.register import CRegister
from lccfq_lang.backend import QPU

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG = str(DATA_DIR / "testing.toml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qpu_mach():
    return QPU(filename=CONFIG, last_pass="mach_optimized")


@pytest.fixture
def qpu_arch():
    return QPU(filename=CONFIG, last_pass="arch_optimized")


@pytest.fixture
def qpu_parsed():
    return QPU(filename=CONFIG, last_pass="parsed")


def _bell(c, qpu):
    c >> qpu.isa.h(tg=0)
    c >> qpu.isa.cx(ct=0, tg=1)
    c >> qpu.isa.measure(tgs=[0, 1])


# ---------------------------------------------------------------------------
# report=False (default) — backward compat
# ---------------------------------------------------------------------------

def test_report_default_is_none(qpu_mach):
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=1) as c:
        _bell(c, qpu)
    assert c.opt_report is None


def test_existing_no_report_kwarg_works(qpu_mach):
    """Phase 0..4 call sites with no `report=` must still work."""
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1) as c:
        _bell(c, qpu)
    assert hasattr(c, "opt_report")
    assert c.opt_report is None


# ---------------------------------------------------------------------------
# report=True — shape
# ---------------------------------------------------------------------------

def test_report_true_populates_dict(qpu_mach):
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    assert isinstance(rep, dict)
    assert set(rep.keys()) == {
        "opt_level", "opt_passes", "routing_strategy",
        "last_pass", "groups", "totals",
    }
    assert rep["opt_level"] == 2
    assert rep["opt_passes"] is None
    assert rep["routing_strategy"] in ("trivial", "sabre_lite", "sabre_fast", "identity")
    assert rep["last_pass"] == "mach_optimized"


def test_report_groups_match_records(qpu_mach):
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    expected = {r.group_name for r in c._opt_records}
    assert {g["name"] for g in rep["groups"]} == expected
    # Per-group pass count matches records
    for g in rep["groups"]:
        from_records = [r for r in c._opt_records
                        if r.group_name == g["name"]]
        assert len(g["passes"]) == len(from_records)


def test_report_totals_sum_seconds(qpu_mach):
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    expected_total = sum(r.delta_seconds for r in c._opt_records)
    assert math.isclose(
        rep["totals"]["total_seconds"],
        expected_total,
        rel_tol=0.0, abs_tol=1e-12,
    )


def test_report_scalarized_delta_consistent(qpu_mach):
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    t = rep["totals"]
    expected = t["cost_before"]["scalarized"] - t["cost_after"]["scalarized"]
    assert math.isclose(t["scalarized_delta"], expected,
                        rel_tol=0.0, abs_tol=1e-9)


def test_report_is_json_serializable(qpu_mach):
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2,
                 report=True) as c:
        _bell(c, qpu)
    json.dumps(c.opt_report)  # must not raise


# ---------------------------------------------------------------------------
# Empty pipeline (last_pass="parsed")
# ---------------------------------------------------------------------------

def test_report_empty_pipeline(qpu_parsed):
    qpu = qpu_parsed
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=0,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    assert rep["last_pass"] == "parsed"
    assert rep["groups"] == []
    t = rep["totals"]
    assert t["cost_before"] == t["cost_after"]
    assert t["scalarized_delta"] == 0.0
    assert t["total_seconds"] == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["yes", 1, 0, None, []])
def test_report_kwarg_must_be_bool(qpu_mach, bad):
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with pytest.raises(TypeError, match="report must be a bool"):
        Circuit(qreg, creg, qpu, shots=1, report=bad)


# ---------------------------------------------------------------------------
# Per-level smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_report_per_level(qpu_mach, level):
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=level,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    assert rep["opt_level"] == level
    if level == 0:
        assert not any(g["name"] == "arch_opt" for g in rep["groups"])
        assert not any(g["name"] == "mach_opt" for g in rep["groups"])
    else:
        names = [g["name"] for g in rep["groups"]]
        assert "arch_opt" in names
        assert "mach_opt" in names
        # arch_opt is fixpoint
        arch = next(g for g in rep["groups"] if g["name"] == "arch_opt")
        assert arch["mode"] == "fixpoint"


# ---------------------------------------------------------------------------
# Perf #1 — per-pass depth=None and group-level depth invariants
# ---------------------------------------------------------------------------

def test_report_group_boundaries_have_int_depth(qpu_mach):
    """groups[i].cost_before/cost_after must always have an int 'depth' field."""
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    for g in rep["groups"]:
        assert g["cost_before"]["depth"] is not None, (
            f"Group {g['name']}: cost_before depth is None"
        )
        assert isinstance(g["cost_before"]["depth"], int), (
            f"Group {g['name']}: cost_before depth not int: {g['cost_before']['depth']!r}"
        )
        assert g["cost_after"]["depth"] is not None, (
            f"Group {g['name']}: cost_after depth is None"
        )
        assert isinstance(g["cost_after"]["depth"], int), (
            f"Group {g['name']}: cost_after depth not int: {g['cost_after']['depth']!r}"
        )


def test_report_totals_have_int_depth(qpu_mach):
    """totals.cost_before/cost_after always have int 'depth' (pipeline-level full Cost)."""
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    t = rep["totals"]
    assert t["cost_before"]["depth"] is not None
    assert isinstance(t["cost_before"]["depth"], int)
    assert t["cost_after"]["depth"] is not None
    assert isinstance(t["cost_after"]["depth"], int)


def test_report_per_pass_depth_none_for_fixpoint_inner(qpu_mach):
    """For fixpoint groups, every per-pass cost_before/cost_after has depth=None."""
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=1,
                 report=True) as c:
        _bell(c, qpu)
    rep = c.opt_report
    for g in rep["groups"]:
        if g["mode"] == "fixpoint":
            for p in g["passes"]:
                assert p["cost_before"]["depth"] is None, (
                    f"Fixpoint group {g['name']} pass {p['name']}: "
                    f"expected depth=None, got {p['cost_before']['depth']!r}"
                )
                assert p["cost_after"]["depth"] is None, (
                    f"Fixpoint group {g['name']} pass {p['name']}: "
                    f"expected depth=None, got {p['cost_after']['depth']!r}"
                )


def test_report_is_still_json_serializable_with_none_depths(qpu_mach):
    """None depth values must JSON-encode without error (null in JSON)."""
    qpu = qpu_mach
    qreg = qpu.qregister(2)
    creg = CRegister(2)
    with Circuit(qreg, creg, qpu, shots=1, opt_level=2,
                 report=True) as c:
        _bell(c, qpu)
    serialized = json.dumps(c.opt_report)  # must not raise
    parsed = json.loads(serialized)
    # Confirm per-pass nulls survived the round-trip
    for g in parsed["groups"]:
        if g["mode"] == "fixpoint":
            for p in g["passes"]:
                assert p["cost_before"]["depth"] is None
                assert p["cost_after"]["depth"] is None
