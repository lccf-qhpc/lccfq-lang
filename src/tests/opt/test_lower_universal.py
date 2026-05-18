"""
Filename: test_lower_universal.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for LowerU2, LowerU3, LowerCU, and FanoutMeasure passes, plus
    integration tests for the lower_expand PassGroup.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from itertools import chain
from pathlib import Path

import numpy as np
import pytest

from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.isa import ISA
from lccfq_lang.backend import QPU
from lccfq_lang.opt.builtin import (
    LowerU2,
    LowerU3,
    LowerCU,
    FanoutMeasure,
    build_lowering_groups,
    LOWERING_STAGES,
)
from lccfq_lang.opt.builtin.lower_passes import slice_groups_for
from lccfq_lang.opt.manager import PassGroup, PassManager
from lccfq_lang.opt.pass_base import PassContext


DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG = str(DATA_DIR / "testing.toml")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qpu():
    return QPU(filename=CONFIG)


@pytest.fixture
def isa():
    return ISA("test")


@pytest.fixture
def qreg(qpu):
    return qpu.qregister(4)


@pytest.fixture
def ctx(qpu, qreg):
    return PassContext(isa=qpu.isa, mapping=qreg.mapping)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instr_tuple(instr: Instruction):
    """Return a comparable tuple of the instruction's key fields."""
    return (
        instr.symbol,
        instr.target_qubits,
        instr.control_qubits,
        instr.params,
        instr.is_controlled,
    )


# ---------------------------------------------------------------------------
# LowerU2 tests
# ---------------------------------------------------------------------------

class TestLowerU2:
    def test_lower_u2_passthrough_no_match(self, qpu, ctx):
        instr = qpu.isa.x(tg=0)
        program = [instr]
        pass_ = LowerU2(qpu.isa)
        result, _ = pass_.run(program, ctx)
        assert result[0] is program[0]

    def test_lower_u2_passthrough_empty_program(self, qpu, ctx):
        pass_ = LowerU2(qpu.isa)
        result, _ = pass_.run([], ctx)
        assert result == []
        assert isinstance(result, list)

    def test_lower_u2_decomposes_canonical_input(self, qpu, qreg, ctx):
        instr = Instruction(symbol="u2", target_qubits=[0], params=[0.5, 1.0])
        program = [instr]
        pass_ = LowerU2(qpu.isa)
        result, _ = pass_.run(program, ctx)
        legacy = qreg.expand(instr)
        assert len(result) == len(legacy)
        for got, expected in zip(result, legacy):
            assert _instr_tuple(got) == _instr_tuple(expected)

    def test_lower_u2_does_not_mutate_input(self, qpu, ctx):
        instr = Instruction(symbol="u2", target_qubits=[0], params=[0.5, 1.0])
        program = [instr]
        original_ids = [id(x) for x in program]
        pass_ = LowerU2(qpu.isa)
        pass_.run(program, ctx)
        assert [id(x) for x in program] == original_ids


# ---------------------------------------------------------------------------
# LowerU3 tests
# ---------------------------------------------------------------------------

class TestLowerU3:
    def test_lower_u3_passthrough_no_match(self, qpu, ctx):
        instr = qpu.isa.h(tg=0)
        program = [instr]
        pass_ = LowerU3(qpu.isa)
        result, _ = pass_.run(program, ctx)
        assert result[0] is program[0]

    def test_lower_u3_passthrough_empty_program(self, qpu, ctx):
        pass_ = LowerU3(qpu.isa)
        result, _ = pass_.run([], ctx)
        assert result == []
        assert isinstance(result, list)

    def test_lower_u3_decomposes_canonical_input(self, qpu, qreg, ctx):
        instr = Instruction(symbol="u3", target_qubits=[0], params=[0.1, 0.2, 0.3])
        program = [instr]
        pass_ = LowerU3(qpu.isa)
        result, _ = pass_.run(program, ctx)
        legacy = qreg.expand(instr)
        assert len(result) == len(legacy)
        for got, expected in zip(result, legacy):
            assert _instr_tuple(got) == _instr_tuple(expected)

    def test_lower_u3_does_not_mutate_input(self, qpu, ctx):
        instr = Instruction(symbol="u3", target_qubits=[0], params=[0.1, 0.2, 0.3])
        program = [instr]
        original_ids = [id(x) for x in program]
        pass_ = LowerU3(qpu.isa)
        pass_.run(program, ctx)
        assert [id(x) for x in program] == original_ids


# ---------------------------------------------------------------------------
# LowerCU tests
# ---------------------------------------------------------------------------

class TestLowerCU:
    def test_lower_cu_passthrough_no_match(self, qpu, ctx):
        instr = qpu.isa.cx(ct=0, tg=1)
        program = [instr]
        pass_ = LowerCU(qpu.isa)
        result, _ = pass_.run(program, ctx)
        assert result[0] is program[0]

    def test_lower_cu_passthrough_empty_program(self, qpu, ctx):
        pass_ = LowerCU(qpu.isa)
        result, _ = pass_.run([], ctx)
        assert result == []
        assert isinstance(result, list)

    def test_lower_cu_decomposes_canonical_input(self, qpu, qreg, ctx):
        instr = Instruction(
            symbol="cu",
            is_controlled=True,
            target_qubits=[1],
            control_qubits=[0],
            params=[0.1, 0.2, 0.3],
        )
        program = [instr]
        pass_ = LowerCU(qpu.isa)
        result, _ = pass_.run(program, ctx)
        legacy = qreg.expand(instr)
        assert len(result) == len(legacy)
        for got, expected in zip(result, legacy):
            assert _instr_tuple(got) == _instr_tuple(expected)

    def test_lower_cu_does_not_mutate_input(self, qpu, ctx):
        instr = Instruction(
            symbol="cu",
            is_controlled=True,
            target_qubits=[1],
            control_qubits=[0],
            params=[0.1, 0.2, 0.3],
        )
        program = [instr]
        original_ids = [id(x) for x in program]
        pass_ = LowerCU(qpu.isa)
        pass_.run(program, ctx)
        assert [id(x) for x in program] == original_ids


# ---------------------------------------------------------------------------
# FanoutMeasure tests
# ---------------------------------------------------------------------------

class TestFanoutMeasure:
    def test_fanout_measure_passthrough_no_match(self, qpu, ctx):
        instr = qpu.isa.x(tg=0)
        program = [instr]
        pass_ = FanoutMeasure(qpu.isa)
        result, _ = pass_.run(program, ctx)
        assert result[0] is program[0]

    def test_fanout_measure_passthrough_empty_program(self, qpu, ctx):
        pass_ = FanoutMeasure(qpu.isa)
        result, _ = pass_.run([], ctx)
        assert result == []
        assert isinstance(result, list)

    def test_fanout_measure_decomposes_canonical_input(self, qpu, qreg, ctx):
        instr = qpu.isa.measure(tgs=[0, 1, 2])
        program = [instr]
        pass_ = FanoutMeasure(qpu.isa)
        result, _ = pass_.run(program, ctx)
        legacy = qreg.expand(instr)
        assert len(result) == len(legacy)
        for got, expected in zip(result, legacy):
            assert _instr_tuple(got) == _instr_tuple(expected)

    def test_fanout_measure_does_not_mutate_input(self, qpu, ctx):
        instr = qpu.isa.measure(tgs=[0, 1, 2])
        program = [instr]
        original_ids = [id(x) for x in program]
        pass_ = FanoutMeasure(qpu.isa)
        pass_.run(program, ctx)
        assert [id(x) for x in program] == original_ids

    def test_fanout_measure_single_target_unchanged_ref(self, qpu, ctx):
        """A single-target measure must be forwarded as the original object reference."""
        instr = qpu.isa.measure(tgs=[0])
        program = [instr]
        pass_ = FanoutMeasure(qpu.isa)
        result, _ = pass_.run(program, ctx)
        assert len(result) == 1
        assert result[0] is program[0]


# ---------------------------------------------------------------------------
# Integration tests — lower_expand group via PassManager
# ---------------------------------------------------------------------------

class TestLowerExpandGroup:
    def test_lower_expand_group_byte_identical_to_legacy_expand(self, qpu, qreg):
        """PassManager(lower_expand) output is element-equal to legacy expand for all cases."""
        u2_instr = Instruction(symbol="u2", target_qubits=[0], params=[0.5, 1.0])
        u3_instr = Instruction(symbol="u3", target_qubits=[0], params=[0.1, 0.2, 0.3])
        cu_instr = Instruction(
            symbol="cu",
            is_controlled=True,
            target_qubits=[1],
            control_qubits=[0],
            params=[0.1, 0.2, 0.3],
        )
        measure_multi = qpu.isa.measure(tgs=[0, 1, 2])
        x_instr = qpu.isa.x(tg=0)

        program = [u2_instr, u3_instr, cu_instr, measure_multi, x_instr]

        # Legacy ground truth
        legacy_out = list(chain.from_iterable(map(qreg.expand, program)))

        # PassManager with lower_expand group
        all_groups = build_lowering_groups(qreg, qpu)
        # lower_expand is index 2 (LOWERING_STAGES index for "expanded")
        expand_idx = LOWERING_STAGES.index("expanded")
        expand_group = all_groups[expand_idx]
        pm = PassManager([expand_group])
        ctx = PassContext(isa=qpu.isa, mapping=qreg.mapping)
        result, _, _ = pm.run(program, ctx)

        assert len(result) == len(legacy_out)
        for got, expected in zip(result, legacy_out):
            assert _instr_tuple(got) == _instr_tuple(expected)

    def test_lower_expand_group_telemetry_has_four_records(self, qpu, qreg):
        """PassManager with lower_expand group emits exactly 4 telemetry records."""
        u2_instr = Instruction(symbol="u2", target_qubits=[0], params=[0.5, 1.0])
        u3_instr = Instruction(symbol="u3", target_qubits=[0], params=[0.1, 0.2, 0.3])
        cu_instr = Instruction(
            symbol="cu",
            is_controlled=True,
            target_qubits=[1],
            control_qubits=[0],
            params=[0.1, 0.2, 0.3],
        )
        measure_multi = qpu.isa.measure(tgs=[0, 1, 2])
        x_instr = qpu.isa.x(tg=0)

        program = [u2_instr, u3_instr, cu_instr, measure_multi, x_instr]

        all_groups = build_lowering_groups(qreg, qpu)
        expand_idx = LOWERING_STAGES.index("expanded")
        expand_group = all_groups[expand_idx]
        pm = PassManager([expand_group])
        ctx = PassContext(isa=qpu.isa, mapping=qreg.mapping)
        _, records, _ = pm.run(program, ctx)

        lower_expand_records = [r for r in records if r.group_name == "lower_expand"]
        assert len(lower_expand_records) == 4

        pass_names = {r.pass_name for r in lower_expand_records}
        assert pass_names == {"lower_u2", "lower_u3", "lower_cu", "fanout_measure"}
