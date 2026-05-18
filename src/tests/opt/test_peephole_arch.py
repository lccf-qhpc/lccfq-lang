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
