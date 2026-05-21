"""
Filename: test_peephole_arch.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for the five arch-level peephole passes: RemoveIdentity,
    CancelInverses, MergeRotations, FuseEulerZYZ, CommuteThroughControl.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import math
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.opt.pass_base import PassContext
from lccfq_lang.opt.builtin.peephole_arch import (
    RemoveIdentity,
    CancelInverses,
    MergeRotations,
    FuseEulerZYZ,
    CommuteThroughControl,
)
from tests.opt._equiv import assert_equivalent

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

isa = ISA("lccfq")
ctx = PassContext(isa=isa)


def _instr_tuple(instr):
    """Return a comparable tuple for structural equality."""
    from lccfq_lang.opt.op_view import OpView
    v = OpView(instr)
    return (v.symbol, v.controls, v.targets, v.params)


# ===========================================================================
# RemoveIdentity
# ===========================================================================

class TestRemoveIdentity:
    def setup_method(self):
        self.pass_ = RemoveIdentity(isa)

    def test_passthrough_empty(self):
        assert self.pass_.run([], ctx)[0] == []

    def test_drops_nop(self):
        program = [isa.nop(tgs=[0]), isa.x(tg=0)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 1
        assert result[0].symbol == "x"

    def test_drops_zero_rotation(self):
        program = [isa.rz(tg=0, params=[0.0])]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_drops_2pi_rotation(self):
        program = [isa.rx(tg=0, params=[2.0 * math.pi])]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_keeps_nontrivial_rotation(self):
        program = [isa.ry(tg=0, params=[math.pi / 4])]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 1
        assert result[0].symbol == "ry"

    def test_does_not_mutate(self):
        program = [isa.nop(tgs=[0]), isa.rz(tg=0, params=[0.0]), isa.x(tg=0)]
        original_ids = [id(i) for i in program]
        self.pass_.run(program, ctx)
        assert [id(i) for i in program] == original_ids

    def test_preserves_semantics(self):
        """A program with zero-angle rotations should be equivalent after removal.

        Note: nop is not supported by _sim.py; use only zero-angle rotations here
        to avoid NotImplementedError from the simulator.
        """
        p_in = [
            isa.h(tg=0),
            isa.rz(tg=0, params=[0.0]),
            isa.rx(tg=0, params=[2.0 * math.pi]),
            isa.cx(ct=0, tg=1),
        ]
        p_out, _ = self.pass_.run(p_in, ctx)
        assert_equivalent(p_in, p_out, 2)


# ===========================================================================
# CancelInverses
# ===========================================================================

class TestCancelInverses:
    def setup_method(self):
        self.pass_ = CancelInverses(isa)

    def test_passthrough_empty(self):
        assert self.pass_.run([], ctx)[0] == []

    def test_xx_cancels(self):
        program = [isa.x(tg=0), isa.x(tg=0)]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_hh_cancels(self):
        program = [isa.h(tg=0), isa.h(tg=0)]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_cx_same_role_cancels(self):
        program = [isa.cx(ct=0, tg=1), isa.cx(ct=0, tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_cx_different_role_does_not_cancel(self):
        program = [isa.cx(ct=0, tg=1), isa.cx(ct=1, tg=0)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 2

    def test_swap_symmetric_cancels(self):
        program = [isa.swap(tg_a=0, tg_b=1), isa.swap(tg_a=1, tg_b=0)]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_s_sdg_cancels(self):
        # Structural assertion only — s/sdg not simulatable
        program = [isa.s(tg=0), isa.sdg(tg=0)]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_t_tdg_cancels(self):
        # Structural assertion only — t/tdg not simulatable
        program = [isa.t(tg=0), isa.tdg(tg=0)]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_non_adjacent_with_non_overlapping_gate_still_cancels(self):
        # h(0), x(1), h(0) — x(1) does not touch qubit 0; h pair can still cancel
        program = [isa.h(tg=0), isa.x(tg=1), isa.h(tg=0)]
        result, _ = self.pass_.run(program, ctx)
        # The two H's cancel; only x(1) remains
        assert len(result) == 1
        assert result[0].symbol == "x"

    def test_non_adjacent_with_blocker_does_not_cancel(self):
        # cx(0,1), x(1), cx(0,1) — x(1) touches qubit 1 which is in the cx union
        program = [isa.cx(ct=0, tg=1), isa.x(tg=1), isa.cx(ct=0, tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 3

    def test_does_not_mutate(self):
        program = [isa.h(tg=0), isa.h(tg=0)]
        original_ids = [id(i) for i in program]
        self.pass_.run(program, ctx)
        assert [id(i) for i in program] == original_ids

    def test_preserves_semantics(self):
        p_in = [
            isa.h(tg=0),
            isa.cx(ct=0, tg=1),
            isa.cx(ct=0, tg=1),
            isa.h(tg=0),
        ]
        p_out, _ = self.pass_.run(p_in, ctx)
        assert_equivalent(p_in, p_out, 2)


# ===========================================================================
# MergeRotations
# ===========================================================================

class TestMergeRotations:
    def setup_method(self):
        self.pass_ = MergeRotations(isa)

    def test_passthrough_empty(self):
        assert self.pass_.run([], ctx)[0] == []

    def test_two_rz_merge(self):
        program = [isa.rz(tg=0, params=[0.3]), isa.rz(tg=0, params=[0.4])]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 1
        assert result[0].symbol == "rz"
        assert abs(result[0].params[0] - 0.7) < 1e-9

    def test_zero_after_merge_drops(self):
        program = [isa.rz(tg=0, params=[math.pi]), isa.rz(tg=0, params=[math.pi])]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_different_axes_no_merge(self):
        program = [isa.rz(tg=0, params=[0.3]), isa.rx(tg=0, params=[0.4])]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 2

    def test_different_qubits_no_merge(self):
        program = [isa.rz(tg=0, params=[0.3]), isa.rz(tg=1, params=[0.4])]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 2

    def test_blocker_prevents_merge(self):
        program = [
            isa.rz(tg=0, params=[0.3]),
            isa.cx(ct=0, tg=1),
            isa.rz(tg=0, params=[0.4]),
        ]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 3

    def test_chain_of_three(self):
        # Left-to-right merge within single pass: first two merge then the
        # merged result merges with the third.
        program = [
            isa.rz(tg=0, params=[0.1]),
            isa.rz(tg=0, params=[0.2]),
            isa.rz(tg=0, params=[0.3]),
        ]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 1
        assert result[0].symbol == "rz"
        assert abs(result[0].params[0] - 0.6) < 1e-9

    def test_does_not_mutate(self):
        program = [isa.rz(tg=0, params=[0.3]), isa.rz(tg=0, params=[0.4])]
        original_ids = [id(i) for i in program]
        self.pass_.run(program, ctx)
        assert [id(i) for i in program] == original_ids

    def test_preserves_semantics(self):
        p_in = [
            isa.rz(tg=0, params=[0.3]),
            isa.rz(tg=0, params=[0.4]),
            isa.cx(ct=0, tg=1),
        ]
        p_out, _ = self.pass_.run(p_in, ctx)
        assert_equivalent(p_in, p_out, 2)


# ===========================================================================
# FuseEulerZYZ
# ===========================================================================

class TestFuseEulerZYZ:
    def setup_method(self):
        self.pass_ = FuseEulerZYZ(isa)

    def test_passthrough_empty(self):
        assert self.pass_.run([], ctx)[0] == []

    def test_identity_triplet_drops(self):
        # rz(0.5) ry(0) rz(-0.5) = identity
        program = [
            isa.rz(tg=0, params=[0.5]),
            isa.ry(tg=0, params=[0.0]),
            isa.rz(tg=0, params=[-0.5]),
        ]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_normalises_angles(self):
        # 3*pi normalises to -pi (canonical); pi/4 stays; check just that result
        # has 3 ops with normalised angles
        program = [
            isa.rz(tg=0, params=[3 * math.pi]),
            isa.ry(tg=0, params=[math.pi / 2]),
            isa.rz(tg=0, params=[math.pi / 4]),
        ]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 3
        from lccfq_lang.opt.builtin._arith import MOD_2PI
        # Each angle should be within (-pi, pi]
        for r in result:
            angle = r.params[0]
            assert -math.pi < angle <= math.pi + 1e-12

    def test_no_match_rz_rx_rz(self):
        # rz rx rz — middle is rx not ry, no match
        program = [
            isa.rz(tg=0, params=[0.5]),
            isa.rx(tg=0, params=[0.3]),
            isa.rz(tg=0, params=[0.2]),
        ]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 3

    def test_blocker_prevents_fuse(self):
        # rz ry cx rz — cx breaks the consecutive triplet on qubit 0
        program = [
            isa.rz(tg=0, params=[0.5]),
            isa.ry(tg=0, params=[0.3]),
            isa.cx(ct=0, tg=1),
            isa.rz(tg=0, params=[-0.5]),
        ]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 4

    def test_does_not_mutate(self):
        program = [
            isa.rz(tg=0, params=[0.5]),
            isa.ry(tg=0, params=[0.0]),
            isa.rz(tg=0, params=[-0.5]),
        ]
        original_ids = [id(i) for i in program]
        self.pass_.run(program, ctx)
        assert [id(i) for i in program] == original_ids

    def test_preserves_semantics(self):
        p_in = [
            isa.rz(tg=0, params=[0.7]),
            isa.ry(tg=0, params=[0.4]),
            isa.rz(tg=0, params=[0.2]),
            isa.cx(ct=0, tg=1),
        ]
        p_out, _ = self.pass_.run(p_in, ctx)
        assert_equivalent(p_in, p_out, 2)


# ===========================================================================
# CommuteThroughControl
# ===========================================================================

class TestCommuteThroughControl:
    def setup_method(self):
        self.pass_ = CommuteThroughControl(isa)

    def test_passthrough_empty(self):
        assert self.pass_.run([], ctx)[0] == []

    def test_commute_rz_past_cx_control(self):
        t = 0.5
        program = [isa.rz(tg=0, params=[t]), isa.cx(ct=0, tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 2
        assert result[0].symbol == "cx"
        assert result[1].symbol == "rz"

    def test_commute_rx_past_cx_target(self):
        t = 0.5
        program = [isa.rx(tg=1, params=[t]), isa.cx(ct=0, tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 2
        assert result[0].symbol == "cx"
        assert result[1].symbol == "rx"

    def test_commute_rz_past_cz_control(self):
        t = 0.5
        program = [isa.rz(tg=0, params=[t]), isa.cz(ct=0, tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 2
        assert result[0].symbol == "cz"
        assert result[1].symbol == "rz"

    def test_commute_rz_past_cz_target(self):
        t = 0.5
        program = [isa.rz(tg=1, params=[t]), isa.cz(ct=0, tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 2
        assert result[0].symbol == "cz"
        assert result[1].symbol == "rz"

    def test_no_commute_rz_on_cx_target(self):
        t = 0.5
        program = [isa.rz(tg=1, params=[t]), isa.cx(ct=0, tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert result[0].symbol == "rz"
        assert result[1].symbol == "cx"

    def test_no_commute_rx_on_cx_control(self):
        t = 0.5
        program = [isa.rx(tg=0, params=[t]), isa.cx(ct=0, tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert result[0].symbol == "rx"
        assert result[1].symbol == "cx"

    def test_does_not_mutate(self):
        program = [isa.rz(tg=0, params=[0.5]), isa.cx(ct=0, tg=1)]
        original_ids = [id(i) for i in program]
        self.pass_.run(program, ctx)
        assert [id(i) for i in program] == original_ids

    def test_preserves_semantics(self):
        t = 0.3
        p_in = [
            isa.rz(tg=0, params=[t]),
            isa.cx(ct=0, tg=1),
        ]
        p_out, _ = self.pass_.run(p_in, ctx)
        assert_equivalent(p_in, p_out, 2)

    # --- DAG-based new tests (Task #76) ---

    def test_commute_rz_past_cx_with_intervening_rotation_on_other_qubit(self):
        """rz(0) and cx(0,1) are not list-adjacent but ARE per-qubit-chain adjacent.

        The intervening ry(1) touches only qubit 1, not qubit 0, so on qubit 0's
        chain the pair is (rz(0), cx(0,1)) which is eligible.  The DAG-based pass
        must commute them even though they are separated in the flat list.
        """
        p_in = [
            isa.rz(tg=0, params=[0.3]),
            isa.ry(tg=1, params=[0.4]),
            isa.cx(ct=0, tg=1),
        ]
        p_out, changed = self.pass_.run(p_in, ctx)
        assert changed is True
        # cx must appear before rz in the output
        symbols = [instr.symbol for instr in p_out]
        assert symbols.index("cx") < symbols.index("rz")
        assert_equivalent(p_in, p_out, 2)

    def test_commute_rx_past_cx_with_intervening_rotation_on_other_qubit(self):
        """rx(1) and cx(0,1) are not list-adjacent but ARE per-qubit-chain adjacent.

        The intervening rz(0) touches only qubit 0, not qubit 1, so on qubit 1's
        chain the pair is (rx(1), cx(0,1)) which is eligible (rx on target).
        """
        p_in = [
            isa.rx(tg=1, params=[0.3]),
            isa.rz(tg=0, params=[0.4]),
            isa.cx(ct=0, tg=1),
        ]
        p_out, changed = self.pass_.run(p_in, ctx)
        assert changed is True
        symbols = [instr.symbol for instr in p_out]
        assert symbols.index("cx") < symbols.index("rx")
        assert_equivalent(p_in, p_out, 2)

    def test_enables_merge_via_fixpoint(self):
        """CommuteThroughControl then MergeRotations collapses rz(0.3) cx rz(0.4)
        into cx rz(0.7).

        After commute: [cx(0,1), rz(0, 0.3), rz(0, 0.4)].
        After merge:   [cx(0,1), rz(0, 0.7)].
        """
        p_in = [
            isa.rz(tg=0, params=[0.3]),
            isa.cx(ct=0, tg=1),
            isa.rz(tg=0, params=[0.4]),
        ]
        commuted, changed = self.pass_.run(p_in, ctx)
        assert changed is True

        merge_pass = MergeRotations(isa)
        merged, _ = merge_pass.run(commuted, ctx)

        assert len(merged) == 2
        assert merged[0].symbol == "cx"
        assert merged[1].symbol == "rz"
        assert abs(merged[1].params[0] - 0.7) < 1e-9

    def test_oscillation_bounded(self):
        """Two successive eligible controls: rz commutes past each cx in turn,
        one per run() call (the i+=2 guard prevents double-commute in one call).
        After two runs, rz is past both cx gates and the third run is idle.
        """
        p_in = [
            isa.rz(tg=0, params=[0.3]),
            isa.cx(ct=0, tg=1),
            isa.cx(ct=0, tg=2),
        ]
        # First run: rz commutes past cx(0,1).
        p_out1, changed1 = self.pass_.run(p_in, ctx)
        assert changed1 is True
        symbols1 = [instr.symbol for instr in p_out1]
        rz_idx1 = symbols1.index("rz")
        # At least one cx must come before rz after the first run.
        cx_indices1 = [i for i, s in enumerate(symbols1) if s == "cx"]
        assert any(cx_i < rz_idx1 for cx_i in cx_indices1)

        # Second run: rz commutes past the remaining cx(0,2).
        p_out2, changed2 = self.pass_.run(p_out1, ctx)
        assert changed2 is True
        symbols2 = [instr.symbol for instr in p_out2]
        rz_idx2 = symbols2.index("rz")
        cx_indices2 = [i for i, s in enumerate(symbols2) if s == "cx"]
        # After two runs, rz must be after all cx gates.
        assert all(cx_i < rz_idx2 for cx_i in cx_indices2)

        # Third run: no eligible (rotation, control) pair remains — idle.
        _, changed3 = self.pass_.run(p_out2, ctx)
        assert changed3 is False

    def test_swap_blocks_commute(self):
        """SWAP(0,1) sits between rz(0) and cx(0,1) on qubit 0's chain.

        _can_commute_forward(rz, swap) is False, so the pair is blocked
        and the output order is unchanged (changed=False).
        """
        p_in = [
            isa.rz(tg=0, params=[0.3]),
            isa.swap(tg_a=0, tg_b=1),
            isa.cx(ct=0, tg=1),
        ]
        p_out, changed = self.pass_.run(p_in, ctx)
        assert changed is False
        assert [instr.symbol for instr in p_out] == ["rz", "swap", "cx"]

    def test_hwea_shape_strictly_fewer_1q_than_level2(self):
        """Level-3 optimization produces strictly fewer 1-qubit ops than level-2
        on a small HWEA-shaped circuit.

        This is the regression that proves opt_level=3 is no longer inert
        after the DAG-based CommuteThroughControl rewrite.

        We use rotations="z" (a single rz per qubit per layer) because that is
        the minimal rotation spec where _can_commute_forward fires on every
        (rz, cx-control) per-qubit adjacent pair and the commuted rz lands
        directly adjacent (on the per-qubit chain) to the next layer's rz,
        allowing MergeRotations to fold them in the next fixpoint iteration.
        Rotation specs whose block ends in a non-rz axis (e.g. "yz", "yzy")
        have an ry blocker between the commuted rz and the next rz, preventing
        the merge; they are not expected to show improvement here.
        """
        import math
        from lccfq_lang.lang.variational import hw_eff_ansatz
        from lccfq_lang.opt.builtin.level_select import passes_for_level, max_iters_for_level
        from lccfq_lang.opt.manager import PassGroup, PassManager
        from lccfq_lang.opt.pass_base import PassContext as PC

        n_qubits = 3
        layers = 2
        rotations = "z"   # rz-only: commuted rz lands adjacent to next layer's rz
        n_params = (layers + 1) * n_qubits * len(rotations)
        params = [math.pi / (i + 1) for i in range(n_params)]
        target = list(range(n_qubits))
        program = hw_eff_ansatz(
            isa, target,
            params=params, layers=layers, rotations=rotations,
        )

        pc = PC(isa=isa)

        def run_level(level: int):
            passes = passes_for_level(level, isa)
            group = PassGroup(
                name="arch_opt",
                mode="fixpoint",
                passes=passes,
                max_iters=max_iters_for_level(level),
            )
            pm = PassManager([group])
            result, _, _ = pm.run(program, pc)
            return result

        # Level 2 needs at least one pass; skip if it would crash on empty passes.
        from lccfq_lang.opt.op_view import OpView as OV

        def count_1q(prog):
            return sum(1 for instr in prog if len(OV(instr).qubits) == 1)

        out2 = run_level(2)
        out3 = run_level(3)

        assert count_1q(out3) < count_1q(out2), (
            f"Expected level-3 to have fewer 1q ops than level-2, "
            f"but got level-2={count_1q(out2)}, level-3={count_1q(out3)}"
        )


# ===========================================================================
# Perf #7 linked-list regression tests — long same-qubit chains
# ===========================================================================

class TestCancelInversesLongChain:
    """16 pairs of h h on q=0 — all cancel, result must be empty.

    Exercises the per-qubit linked-list cancel path through many iterations
    on the same qubit, verifying O(1) splice correctness throughout.
    """

    def setup_method(self):
        self.pass_ = CancelInverses(isa)

    def test_cancel_inverses_long_same_qubit_chain(self):
        # 16 pairs of (h, h): should cancel entirely to [].
        n_pairs = 16
        program = []
        for _ in range(n_pairs):
            program.append(isa.h(tg=0))
            program.append(isa.h(tg=0))
        result, changed = self.pass_.run(program, ctx)
        assert result == [], f"Expected empty result, got {len(result)} ops"
        assert changed is True


class TestMergeRotationsLongChain:
    """32 rz rotations on q=0 whose angles sum to 0 (mod 2*pi) — should drop all.

    Exercises the replace-in-place and final-drop paths of the linked-list
    MergeRotations implementation across a long chain.
    """

    def setup_method(self):
        self.pass_ = MergeRotations(isa)

    def test_merge_rotations_long_same_qubit_chain(self):
        import math
        # 32 rz(pi/16) each; sum = 32 * pi/16 = 2*pi => zero after MOD_2PI.
        n = 32
        angle = (2 * math.pi) / n
        program = [isa.rz(tg=0, params=[angle]) for _ in range(n)]
        result, changed = self.pass_.run(program, ctx)
        assert result == [], f"Expected empty result, got {len(result)} ops"
        assert changed is True


class TestFuseEulerZYZLongChain:
    """8 consecutive (rz, ry, rz) triplets on q=0, each collapsing to identity.

    Each triplet is rz(alpha) ry(0) rz(-alpha) = identity.  The chain
    exercises both the pp_node and p_node cancel path repeatedly, stressing
    the prev_per_qubit pointer updates after each triple drop.
    """

    def setup_method(self):
        self.pass_ = FuseEulerZYZ(isa)

    def test_fuse_euler_zyz_chained_triplets(self):
        import math
        # 8 identity triplets: rz(0.5) ry(0) rz(-0.5) each.
        n_triplets = 8
        program = []
        for _ in range(n_triplets):
            program.append(isa.rz(tg=0, params=[0.5]))
            program.append(isa.ry(tg=0, params=[0.0]))
            program.append(isa.rz(tg=0, params=[-0.5]))
        result, changed = self.pass_.run(program, ctx)
        assert result == [], f"Expected empty result, got {len(result)} ops"
        assert changed is True
