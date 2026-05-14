"""
Filename: test_level_select.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for level_select: passes_for_level, max_iters_for_level,
    ALL_ARCH_PASSES, and resolve_opt_passes.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.opt.builtin.level_select import (
    passes_for_level,
    max_iters_for_level,
    resolve_opt_passes,
    ALL_ARCH_PASSES,
    VALID_OPT_LEVELS,
)
from lccfq_lang.opt.builtin.peephole_arch import (
    RemoveIdentity,
    CancelInverses,
    MergeRotations,
    FuseEulerZYZ,
    CommuteThroughControl,
)
from lccfq_lang.opt.builtin.templates_arch import HCXHRule, SwapElision

isa = ISA("lccfq")


class TestPassesForLevel:
    def test_level_0_returns_empty(self):
        assert passes_for_level(0, isa) == []

    def test_level_1_pass_names(self):
        passes = passes_for_level(1, isa)
        names = [p.name for p in passes]
        assert names == ["remove_identity", "cancel_inverses", "merge_rotations"]

    def test_level_2_pass_names(self):
        passes = passes_for_level(2, isa)
        names = [p.name for p in passes]
        assert names == [
            "remove_identity",
            "cancel_inverses",
            "merge_rotations",
            "fuse_euler_zyz",
            "hcxh_to_cz",
            "swap_elision",
        ]

    def test_level_3_pass_names(self):
        passes = passes_for_level(3, isa)
        names = [p.name for p in passes]
        assert names == [
            "remove_identity",
            "cancel_inverses",
            "merge_rotations",
            "fuse_euler_zyz",
            "hcxh_to_cz",
            "swap_elision",
            "commute_through_control",
        ]
        # commute is last
        assert names[-1] == "commute_through_control"

    def test_invalid_level_4_raises(self):
        with pytest.raises(ValueError, match="opt_level must be one of"):
            passes_for_level(4, isa)

    def test_invalid_level_negative_raises(self):
        with pytest.raises(ValueError, match="opt_level must be one of"):
            passes_for_level(-1, isa)

    def test_invalid_level_string_raises(self):
        with pytest.raises(ValueError, match="opt_level must be one of"):
            passes_for_level("1", isa)


class TestMaxItersForLevel:
    def test_level_0(self):
        assert max_iters_for_level(0) == 1

    def test_level_1(self):
        assert max_iters_for_level(1) == 3

    def test_level_2(self):
        assert max_iters_for_level(2) == 5

    def test_level_3(self):
        assert max_iters_for_level(3) == 10

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="opt_level must be one of"):
            max_iters_for_level(99)


class TestAllArchPasses:
    def test_all_keys_match_class_names(self):
        for key, cls in ALL_ARCH_PASSES.items():
            assert key == cls.name, (
                f"Key {key!r} does not match {cls.__name__}.name = {cls.name!r}"
            )

    def test_contains_expected_passes(self):
        expected = {
            "remove_identity",
            "cancel_inverses",
            "merge_rotations",
            "fuse_euler_zyz",
            "commute_through_control",
            "hcxh_to_cz",
            "swap_elision",
        }
        assert set(ALL_ARCH_PASSES.keys()) == expected


class TestResolveOptPasses:
    def test_known_arch_pass_returns_instance(self):
        arch_passes, mach_passes = resolve_opt_passes(["remove_identity"], isa)
        assert len(arch_passes) == 1
        assert isinstance(arch_passes[0], RemoveIdentity)
        assert mach_passes == []

    def test_multiple_known_arch_passes(self):
        arch_passes, mach_passes = resolve_opt_passes(["remove_identity", "cancel_inverses"], isa)
        assert len(arch_passes) == 2
        assert isinstance(arch_passes[0], RemoveIdentity)
        assert isinstance(arch_passes[1], CancelInverses)
        assert mach_passes == []

    def test_returns_tuple_of_two_lists(self):
        result = resolve_opt_passes(["remove_identity"], isa)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_unknown_pass_raises(self):
        with pytest.raises(ValueError, match="Unknown pass: bogus"):
            resolve_opt_passes(["bogus"], isa)

    def test_non_list_raises_type_error(self):
        with pytest.raises(TypeError, match="names must be list"):
            resolve_opt_passes("not_a_list", isa)

    def test_list_of_non_strings_raises_type_error(self):
        with pytest.raises(TypeError, match="names must be list"):
            resolve_opt_passes([1, 2], isa)
