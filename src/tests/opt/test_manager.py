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
        return list(program), False


class DropOne(Pass):
    """Pass that drops the first element of the program."""
    name = "drop_one"
    applies_to = "arch"

    def run(self, program, ctx):
        changed = len(program) > 0
        return list(program[1:]), changed


class MachNoOp(Pass):
    """NoOp for mach-level programs."""
    name = "mach_noop"
    applies_to = "mach"

    def run(self, program, ctx):
        return list(program), False


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
        cost_after have int depth; middle records may have None depth.

        Perf #4 note: when a pass returns changed=False, cost_after is set to
        cost_before (no measurement). For pass 0, cost_before is from
        _measure (int depth), so cost_after also has int depth. For non-first
        passes, cost_before is from measure_counts (None depth).
        """
        group = PassGroup(name="lin", mode="linear",
                          passes=[NoOp(), NoOp(), NoOp()])
        pm = PassManager([group])
        prog = make_arch_program(3)
        _, records, _ = pm.run(prog)
        assert len(records) == 3
        # First pass pre-cost: full Cost.measure — int depth
        assert records[0].cost_before.depth is not None and records[0].cost_before.depth >= 0
        # Last pass post-cost: full Cost.measure — int depth
        assert records[-1].cost_after.depth is not None and records[-1].cost_after.depth >= 0
        # records[1].cost_before: not first → measure_counts → None depth
        assert records[1].cost_before.depth is None


# ---------------------------------------------------------------------------
# Perf #2 — LRU-1 cost cache across group boundaries
# ---------------------------------------------------------------------------

class TestPerf2CostCache:
    def test_perf2_cache_hits_across_boundaries(self):
        """Multi-group compile produces at least one boundary cache hit."""
        # Two back-to-back linear groups so that group-1 last-pass-post
        # is identical to group-2 first-pass-pre (same program object).
        g1 = PassGroup(name="g1", mode="linear", passes=[NoOp()])
        g2 = PassGroup(name="g2", mode="linear", passes=[NoOp()])
        pm = PassManager([g1, g2])
        pm.run(make_arch_program(3))
        assert pm._cache_stats["hits"] >= 1
        assert pm._cache_stats["misses"] >= 1

    def test_perf2_cache_resets_between_runs(self):
        """`.run()` resets the cache state and stats."""
        group = PassGroup(name="g", mode="linear", passes=[NoOp()])
        pm = PassManager([group])
        pm.run(make_arch_program(3))
        first_stats = dict(pm._cache_stats)
        pm.run(make_arch_program(3))
        # Stats reset to fresh counts (NOT additive across runs).
        # Two identical single-group runs produce the same total measurements.
        assert pm._cache_stats["hits"] + pm._cache_stats["misses"] == \
               first_stats["hits"] + first_stats["misses"]
        # Cache slot itself cleared at entry — stats are not accumulated.
        assert pm._cache_stats == first_stats

    def test_perf2_no_behavior_change(self):
        """Cost values returned to groups_meta are unchanged vs. an uncached
        re-measurement."""
        group = PassGroup(name="fp", mode="fixpoint", passes=[NoOp()])
        pm = PassManager([group])
        prog = make_arch_program(3)
        _, records, groups_meta = pm.run(prog)
        group_in, group_out = groups_meta["fp"]
        # Re-measure independently with the public API; results must match.
        assert group_in == Cost.measure(prog, "arch", None)

    def test_perf2_cache_hit_returns_same_object(self):
        """Cache hit returns the SAME Cost instance, not just an equal one."""
        # Two linear groups → the last-pass-post of g1 is the first-pass-pre
        # of g2 (same program object handed across the boundary).
        g1 = PassGroup(name="g1", mode="linear", passes=[NoOp()])
        g2 = PassGroup(name="g2", mode="linear", passes=[NoOp()])
        pm = PassManager([g1, g2])
        _, records, _ = pm.run(make_arch_program(3))
        # records[0].cost_after (last-pass-post of g1) and
        # records[1].cost_before (first-pass-pre of g2) must be
        # the same Cost instance — the cache returned the cached object.
        assert records[0].cost_after is records[1].cost_before

    def test_perf2_pass_in_place_mutation_assert(self):
        """Sanity: passes return new lists, so id(program) before vs after
        a pass invocation MUST differ — proving the cache cannot false-hit on
        a transformed-but-same-id program.

        This guards against a future regression where a pass mutates in place
        AND returns the same list — which would silently break the cache
        invariant documented in the _measure() docstring.
        """
        class IdentityChecker(Pass):
            name = "id_check"
            applies_to = "arch"
            seen_ids = []

            def run(self, program, ctx):
                IdentityChecker.seen_ids.append(id(program))
                out = list(program)
                IdentityChecker.seen_ids.append(id(out))
                return out, False

        IdentityChecker.seen_ids = []
        group = PassGroup(name="g", mode="linear", passes=[IdentityChecker()])
        pm = PassManager([group])
        pm.run(make_arch_program(3))
        # Input id and output id must differ (pass returns a new list).
        assert IdentityChecker.seen_ids[0] != IdentityChecker.seen_ids[1]


# ---------------------------------------------------------------------------
# Perf #4 — changed-flag signal
# ---------------------------------------------------------------------------

class TestPerf4ChangedFlag:
    """Tests for the Perf #4 changed-flag optimization."""

    def test_perf4_pass_returns_tuple(self):
        """Every concrete in-tree Pass subclass returns a (program, bool) tuple."""
        from lccfq_lang.arch.isa import ISA
        from lccfq_lang.mach.topology import QPUTopology
        from lccfq_lang.sys.base import QPUConfig
        from lccfq_lang.opt.builtin.lower_universal import LowerU2, LowerU3, LowerCU, FanoutMeasure
        from lccfq_lang.opt.builtin.peephole_arch import (
            RemoveIdentity, CancelInverses, MergeRotations, FuseEulerZYZ, CommuteThroughControl
        )
        from lccfq_lang.opt.builtin.templates_arch import HCXHRule, SwapElision
        from lccfq_lang.opt.builtin.peephole_mach import (
            RemoveIdentityMach, MergeAdjacent1Q, EulerXYRecompose
        )
        from lccfq_lang.opt.builtin.scheduling_mach import DeferMeasurement, ParallelizeLayers
        from lccfq_lang.opt.builtin.native_synthesis import RyRzRyToHardware
        from lccfq_lang.opt.builtin.routing import LookaheadSwapInsertion

        _isa = ISA("lccfq")
        _ctx = PassContext(isa=_isa)

        _topo_spec = {
            "qpu": {
                "name": "test",
                "location": "lab",
                "topology": "linear",
                "qubit_count": 2,
                "qubits": [0, 1],
                "couplings": [(0, 1)],
                "exclusions": [],
            },
            "network": {"ip": "127.0.0.1", "port": 1234},
        }
        _topo = QPUTopology(QPUConfig(_topo_spec))

        # All non-routing passes take (isa,); routing takes (None, isa, topology)
        isa_passes = [
            LowerU2, LowerU3, LowerCU, FanoutMeasure,
            RemoveIdentity, CancelInverses, MergeRotations, FuseEulerZYZ, CommuteThroughControl,
            HCXHRule, SwapElision,
            RemoveIdentityMach, MergeAdjacent1Q, EulerXYRecompose,
            DeferMeasurement, ParallelizeLayers, RyRzRyToHardware,
        ]
        for cls in isa_passes:
            inst = cls(_isa)
            result = inst.run([], _ctx)
            assert isinstance(result, tuple) and len(result) == 2, (
                f"{cls.__name__}.run([]) returned {type(result).__name__}, expected 2-tuple"
            )
            out, changed = result
            assert isinstance(out, list), f"{cls.__name__}: first element must be list"
            assert isinstance(changed, bool), f"{cls.__name__}: second element must be bool"

        # LookaheadSwapInsertion (qreg=None, isa, topology)
        lsi = LookaheadSwapInsertion(None, _isa, _topo)
        result = lsi.run([], _ctx)
        assert isinstance(result, tuple) and len(result) == 2
        out, changed = result
        assert isinstance(out, list)
        assert isinstance(changed, bool)

    def test_perf4_changed_false_skips_post_cost(self):
        """A pass returning changed=False causes manager to skip the
        per-pass cost_after measure_counts call (in fixpoint inner loop)."""
        call_count = {"n": 0}
        original_measure_counts = Cost.measure_counts.__func__

        class NoOpChangedFalse(Pass):
            name = "noop_false"
            applies_to = "arch"

            def run(self, program, ctx):
                return list(program), False

        group = PassGroup("g", "fixpoint", [NoOpChangedFalse()], max_iters=5)
        pm = PassManager([group])

        # Monkey-patch Cost.measure_counts to count calls
        import lccfq_lang.opt.cost as cost_mod
        original = cost_mod.Cost.measure_counts
        call_count = [0]

        @classmethod
        def spy_measure_counts(cls, program, kind, qpu_config=None):
            call_count[0] += 1
            return original.__func__(cls, program, kind, qpu_config)

        cost_mod.Cost.measure_counts = spy_measure_counts
        try:
            pm.run(make_arch_program(3))
        finally:
            cost_mod.Cost.measure_counts = original

        # With Perf #4: fixpoint breaks after 1 iteration (iter_changed=False).
        # In that 1 iteration: cost_before is measured (1 call).
        # cost_after is SKIPPED (changed=False → reuse cost_before).
        # Total: 1 measure_counts call (cost_before of the single pass).
        assert call_count[0] == 1, (
            f"Expected 1 measure_counts call (cost_before only), got {call_count[0]}"
        )

    def test_perf4_fixpoint_breaks_on_no_change(self):
        """A fixpoint group of only change-free passes terminates after exactly
        1 iteration, regardless of max_iters."""
        class NoOpChangedFalse(Pass):
            name = "noop_false"
            applies_to = "arch"

            def run(self, program, ctx):
                return list(program), False

        group = PassGroup("g", "fixpoint", [NoOpChangedFalse()], max_iters=10)
        pm = PassManager([group])
        _, records, _ = pm.run(make_arch_program(3))
        # With max_iters=10 but iter_changed=False after iteration 0,
        # we expect exactly 1 record (1 pass × 1 iteration).
        g_records = [r for r in records if r.group_name == "g"]
        assert len(g_records) == 1, (
            f"Expected 1 record (1 iteration), got {len(g_records)}"
        )
        assert g_records[0].iteration == 0

    def test_perf4_fixpoint_breaks_skips_final_group_cost_after_measurement(self):
        """When the outer loop breaks on iter_changed=False, the trailing
        group_cost_after self._measure() is NOT called.
        Verified via _cache_stats: only the initial group_cost_before miss
        should appear; no additional miss for group_cost_after."""
        class NoOpChangedFalse(Pass):
            name = "noop_false"
            applies_to = "arch"

            def run(self, program, ctx):
                return list(program), False

        group = PassGroup("g", "fixpoint", [NoOpChangedFalse()], max_iters=5)
        pm = PassManager([group])
        pm.run(make_arch_program(3))
        # Initial group_cost_before: 1 miss.
        # No group_cost_after (skipped on iter_changed=False break).
        assert pm._cache_stats["misses"] == 1, (
            f"Expected 1 miss (initial group_cost_before only), got {pm._cache_stats['misses']}"
        )
        assert pm._cache_stats["hits"] == 0

    def test_perf4_passrecord_has_changed(self):
        """PassRecord carries the changed flag from the pass."""
        class AlwaysChanged(Pass):
            name = "always_changed"
            applies_to = "arch"

            def run(self, program, ctx):
                return list(program), True

        class NeverChanged(Pass):
            name = "never_changed"
            applies_to = "arch"

            def run(self, program, ctx):
                return list(program), False

        group = PassGroup("g", "linear", [AlwaysChanged(), NeverChanged()])
        pm = PassManager([group])
        _, records, _ = pm.run(make_arch_program(3))
        assert records[0].changed is True
        assert records[1].changed is False

    def test_perf4_changed_false_does_not_skip_in_linear_mode(self):
        """In a LINEAR group, changed=False does NOT skip subsequent passes
        (linear groups always run every pass once). Only the per-pass
        cost_after measure_counts is elided."""
        calls = []

        class TrackedNoop(Pass):
            def __init__(self, tag):
                self.name = tag
                self.applies_to = "arch"

            def run(self, program, ctx):
                calls.append(self.name)
                return list(program), False

        group = PassGroup("g", "linear",
                          [TrackedNoop("a"), TrackedNoop("b"), TrackedNoop("c")])
        pm = PassManager([group])
        pm.run(make_arch_program(3))
        assert calls == ["a", "b", "c"]

    def test_perf4_changed_true_after_changed_false_still_measures(self):
        """In a fixpoint iter, a later pass returning changed=True after
        an earlier pass returning changed=False still produces a valid
        post-cost record and iter_changed=True allows continuation."""
        class NoOpFalse(Pass):
            name = "noop_false"
            applies_to = "arch"

            def run(self, p, ctx):
                return list(p), False

        class RemoveOne(Pass):
            name = "remove_one"
            applies_to = "arch"

            def run(self, p, ctx):
                if p:
                    return list(p[:-1]), True
                return list(p), False

        group = PassGroup("g", "fixpoint", [NoOpFalse(), RemoveOne()], max_iters=5)
        pm = PassManager([group])
        out, records, _ = pm.run(make_arch_program(3))
        # RemoveOne fires on each non-empty iteration; NoOpFalse never changes.
        assert any(r.changed for r in records)
        assert any(not r.changed for r in records)

    def test_perf4_lying_pass_detected_in_debug_mode(self):
        """In debug mode (PassManager.debug_assert_changed=True), a pass that
        returns changed=False but DID modify the program raises AssertionError."""
        class LyingPass(Pass):
            name = "liar"
            applies_to = "arch"

            def run(self, p, ctx):
                # Actually modifies program but claims changed=False.
                return list(p[:-1]) if p else list(p), False

        group = PassGroup("g", "linear", [LyingPass()])
        pm = PassManager([group])
        pm.debug_assert_changed = True
        with pytest.raises(AssertionError, match="changed=False but cost differs"):
            pm.run(make_arch_program(3))
