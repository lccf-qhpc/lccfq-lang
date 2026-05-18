"""
Filename: test_manager.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for PassGroup validation and PassManager.run().

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.mach.ir import Gate
from lccfq_lang.opt import (
    Pass, PassContext, PassRecord, PassGroup, PassManager, Cost
)


# ---------------------------------------------------------------------------
# Concrete pass implementations for testing
# ---------------------------------------------------------------------------

class NoOp(Pass):
    """Pass that returns a copy of the program unchanged."""
    name = "noop"
    applies_to = "arch"

    def run(self, program, ctx):
        return list(program)


class DropOne(Pass):
    """Pass that drops the first element of the program."""
    name = "drop_one"
    applies_to = "arch"

    def run(self, program, ctx):
        return list(program[1:])


class MachNoOp(Pass):
    """NoOp for mach-level programs."""
    name = "mach_noop"
    applies_to = "mach"

    def run(self, program, ctx):
        return list(program)


class NeverCallPass(Pass):
    """Pass that must never be called — raises if run() is invoked."""
    name = "never_call"
    applies_to = "arch"

    def run(self, program, ctx):
        raise AssertionError("NeverCallPass.run() was called — it should not have been")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def instr(symbol, target_qubits=None):
    return Instruction(symbol=symbol, target_qubits=target_qubits)


def make_arch_program(n):
    return [instr(f"op{i}", target_qubits=[i % 3]) for i in range(n)]


# ---------------------------------------------------------------------------
# PassGroup validation
# ---------------------------------------------------------------------------

class TestPassGroupValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            PassGroup(name="", mode="linear", passes=[NoOp()])

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError):
            PassGroup(name="g", mode="waterfall", passes=[NoOp()])

    def test_empty_passes_raises(self):
        with pytest.raises(ValueError):
            PassGroup(name="g", mode="linear", passes=[])

    def test_mixed_applies_to_raises(self):
        with pytest.raises(ValueError, match="mixed applies_to"):
            PassGroup(name="g", mode="linear", passes=[NoOp(), MachNoOp()])

    def test_max_iters_zero_raises(self):
        with pytest.raises(ValueError):
            PassGroup(name="g", mode="fixpoint", passes=[NoOp()], max_iters=0)

    def test_negative_tol_raises(self):
        with pytest.raises(ValueError):
            PassGroup(name="g", mode="fixpoint", passes=[NoOp()], tol=-1.0)

    def test_valid_group_ok(self):
        g = PassGroup(name="valid", mode="linear", passes=[NoOp()])
        assert g.name == "valid"


# ---------------------------------------------------------------------------
# Linear mode
# ---------------------------------------------------------------------------

class TestLinearMode:
    def test_linear_one_pass_one_record(self):
        prog = make_arch_program(3)
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        out, records, _ = pm.run(prog)
        assert len(records) == 1
        assert records[0].iteration == 0
        assert records[0].pass_name == "noop"
        assert records[0].group_name == "g"

    def test_linear_output_unchanged(self):
        prog = make_arch_program(3)
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        out, _, _ = pm.run(prog)
        assert len(out) == len(prog)

    def test_linear_does_not_mutate_input(self):
        prog = make_arch_program(3)
        original_ids = [id(x) for x in prog]
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        pm.run(prog)
        # Original list must still hold the same objects
        assert [id(x) for x in prog] == original_ids

    def test_linear_returns_new_list_object(self):
        """Spec: top-level shallow copy — output list MUST NOT be the input list."""
        prog = make_arch_program(3)
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        out, _, _ = pm.run(prog)
        assert out is not prog

    def test_linear_pass_record_fields(self):
        prog = make_arch_program(2)
        group = PassGroup(name="grp", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        _, records, _ = pm.run(prog)
        rec = records[0]
        assert isinstance(rec.cost_before, Cost)
        assert isinstance(rec.cost_after, Cost)
        assert rec.delta_seconds >= 0.0


# ---------------------------------------------------------------------------
# Fixpoint mode
# ---------------------------------------------------------------------------

class TestFixpointMode:
    def test_fixpoint_noop_terminates_after_one_sweep(self):
        """NoOp produces no improvement → breaks after first sweep."""
        prog = make_arch_program(3)
        group = PassGroup(name="g", mode="fixpoint", passes=[NoOp()], max_iters=5, tol=0.0)
        pm = PassManager([group])
        _, records, _ = pm.run(prog)
        # One pass, one sweep → one record
        assert len(records) == 1
        assert records[0].iteration == 0

    def test_fixpoint_drop_one_records_and_final(self):
        """DropOne shrinks the program by 1 each sweep.
        4-op program, max_iters=5:
          sweep 0: 4->3  improvement > 0 → continue
          sweep 1: 3->2  improvement > 0 → continue
          sweep 2: 2->1  improvement > 0 → continue
          sweep 3: 1->0  improvement > 0 → continue
          sweep 4: 0->0  empty, no run, improvement = 0 → break
        Records: one per sweep × one pass = 5 total.
        """
        prog = make_arch_program(4)
        group = PassGroup(name="g", mode="fixpoint", passes=[DropOne()], max_iters=5, tol=0.0)
        pm = PassManager([group])
        out, records, _ = pm.run(prog)
        assert out == []
        assert len(records) == 5

    def test_fixpoint_iteration_numbers(self):
        """Iteration counter in records reflects the sweep number."""
        prog = make_arch_program(2)
        group = PassGroup(name="g", mode="fixpoint", passes=[DropOne()], max_iters=3, tol=0.0)
        pm = PassManager([group])
        _, records, _ = pm.run(prog)
        iters = [r.iteration for r in records]
        # sweep 0: 2->1 improve, sweep 1: 1->0 improve, sweep 2: 0->0 break → 3 records
        assert iters == [0, 1, 2]


# ---------------------------------------------------------------------------
# Type mismatch
# ---------------------------------------------------------------------------

class TestTypeMismatch:
    def test_mach_group_with_arch_program_raises(self):
        prog = make_arch_program(2)
        group = PassGroup(name="g", mode="fixpoint", passes=[MachNoOp()])
        pm = PassManager([group])
        with pytest.raises(TypeError):
            pm.run(prog)

    def test_arch_group_with_mach_program_raises(self):
        prog = [Gate(symbol="rx", target_qubits=[0], control_qubits=[], params=[0.5])]
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        with pytest.raises(TypeError):
            pm.run(prog)

    def test_type_mismatch_error_message_format(self):
        """Spec: error reads 'PassGroup <name> expects <kind> program, got <Type>'."""
        prog = [Gate(symbol="rx", target_qubits=[0], control_qubits=[], params=[0.5])]
        group = PassGroup(name="grpA", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        with pytest.raises(
            TypeError,
            match=r"PassGroup grpA expects arch program, got Gate",
        ):
            pm.run(prog)



# ---------------------------------------------------------------------------
# Empty program: pass not called
# ---------------------------------------------------------------------------

class TestEmptyProgram:
    def test_empty_prog_linear_not_called(self):
        """Empty program: pass is NOT invoked but a record IS produced.

        For a single-pass linear group, Perf #1 decision C.2 uses full
        Cost.measure for both the first (pre) and last (post) costs.  On an
        empty program, Cost.measure returns Cost(0, 0, 0, 0, None) — depth=0,
        not None — so the equality assertion is unchanged.
        """
        group = PassGroup(name="g", mode="linear", passes=[NeverCallPass()])
        pm = PassManager([group])
        out, records, _ = pm.run([])
        assert out == []
        assert len(records) == 1
        assert records[0].cost_before == Cost(0, 0, 0, 0, None)
        assert records[0].cost_after  == Cost(0, 0, 0, 0, None)

    def test_empty_prog_fixpoint_not_called(self):
        """Empty program in fixpoint: no improvement → breaks after one sweep."""
        group = PassGroup(name="g", mode="fixpoint", passes=[NeverCallPass()], max_iters=5)
        pm = PassManager([group])
        out, records, _ = pm.run([])
        assert out == []
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

class TestContext:
    def test_scratchpad_reset_between_runs(self):
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        ctx = PassContext(scratchpad={"key": "value"})
        pm.run([], ctx)
        assert ctx.scratchpad == {}

    def test_none_ctx_creates_default(self):
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        out, records, _ = pm.run([], ctx=None)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# Perf #1 — new tests for 3-tuple return and depth=None invariants
# ---------------------------------------------------------------------------

class TestPerf1ReturnShape:
    def test_passmanager_run_returns_triple(self):
        """PassManager.run now returns (program, records, groups_meta)."""
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        result = pm.run([])
        assert isinstance(result, tuple)
        assert len(result) == 3
        program, records, groups_meta = result
        assert isinstance(records, list)
        assert isinstance(groups_meta, dict)

    def test_linear_groups_absent_from_groups_meta(self):
        """Linear groups do not appear in groups_meta."""
        group = PassGroup(name="lin", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        _, _, groups_meta = pm.run(make_arch_program(2))
        assert "lin" not in groups_meta

    def test_fixpoint_groups_meta_populated(self):
        """groups_meta is populated for fixpoint groups."""
        group = PassGroup(name="fp", mode="fixpoint", passes=[NoOp()])
        pm = PassManager([group])
        _, _, groups_meta = pm.run(make_arch_program(2))
        assert "fp" in groups_meta
        group_in, group_out = groups_meta["fp"]
        assert isinstance(group_in, Cost)
        assert isinstance(group_out, Cost)

    def test_fixpoint_groups_meta_has_full_depth(self):
        """groups_meta fixpoint entry has int depth (full Cost.measure call)."""
        group = PassGroup(name="fp", mode="fixpoint", passes=[NoOp()])
        pm = PassManager([group])
        _, _, groups_meta = pm.run(make_arch_program(3))
        group_in, group_out = groups_meta["fp"]
        assert group_in.depth is not None and group_in.depth >= 0
        assert group_out.depth is not None and group_out.depth >= 0


class TestPerf1DepthNone:
    def test_fixpoint_inner_records_have_none_depth(self):
        """For fixpoint groups, every per-pass cost_before/cost_after has depth=None."""
        group = PassGroup(name="fp", mode="fixpoint", passes=[NoOp()])
        pm = PassManager([group])
        prog = make_arch_program(3)
        _, records, _ = pm.run(prog)
        for r in records:
            assert r.cost_before.depth is None, (
                f"Expected cost_before.depth=None for fixpoint record, got {r.cost_before.depth}"
            )
            assert r.cost_after.depth is None, (
                f"Expected cost_after.depth=None for fixpoint record, got {r.cost_after.depth}"
            )

    def test_linear_single_pass_has_int_depth(self):
        """Single-pass linear group: both cost_before and cost_after have int depth."""
        group = PassGroup(name="lin", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        _, records, _ = pm.run(make_arch_program(3))
        assert len(records) == 1
        assert records[0].cost_before.depth is not None and records[0].cost_before.depth >= 0
        assert records[0].cost_after.depth  is not None and records[0].cost_after.depth  >= 0

    def test_linear_multi_pass_boundary_has_int_depth(self):
        """Multi-pass linear group: first record's cost_before and last record's
        cost_after have int depth; middle records may have None depth."""
        group = PassGroup(name="lin", mode="linear",
                          passes=[NoOp(), NoOp(), NoOp()])
        pm = PassManager([group])
        prog = make_arch_program(3)
        _, records, _ = pm.run(prog)
        assert len(records) == 3
        # First pass pre-cost: full Cost.measure
        assert records[0].cost_before.depth is not None and records[0].cost_before.depth >= 0
        # Last pass post-cost: full Cost.measure
        assert records[-1].cost_after.depth is not None and records[-1].cost_after.depth >= 0
        # Middle records (pass 0 post, pass 1 pre, pass 1 post, pass 2 pre): measure_counts
        # records[0].cost_after: not last → measure_counts → None
        assert records[0].cost_after.depth is None
        # records[1].cost_before: not first → measure_counts → None
        assert records[1].cost_before.depth is None
