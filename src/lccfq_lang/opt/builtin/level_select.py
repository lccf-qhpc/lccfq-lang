"""
Filename: level_select.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Maps Phase 2 opt_level to a curated arch-pass list and exposes
    ALL_ARCH_PASSES for explicit opt_passes name resolution.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
from typing import List, Type
from lccfq_lang.arch.isa import ISA
from lccfq_lang.opt.pass_base import Pass
from .peephole_arch import (
    RemoveIdentity,
    CancelInverses,
    MergeRotations,
    FuseEulerZYZ,
    CommuteThroughControl,
)
from .templates_arch import HCXHRule, SwapElision

VALID_OPT_LEVELS: tuple[int, ...] = (0, 1, 2, 3)

# Name -> Pass class registry for opt_passes resolution.
# Every value must be constructible as cls(isa).
ALL_ARCH_PASSES: dict[str, Type[Pass]] = {
    RemoveIdentity.name:        RemoveIdentity,
    CancelInverses.name:        CancelInverses,
    MergeRotations.name:        MergeRotations,
    FuseEulerZYZ.name:          FuseEulerZYZ,
    CommuteThroughControl.name: CommuteThroughControl,
    HCXHRule.name:              HCXHRule,
    SwapElision.name:           SwapElision,
}


def passes_for_level(level: int, isa: ISA) -> List[Pass]:
    """Return the list of arch-level passes for the given opt_level.

    :raises ValueError: if level is not in VALID_OPT_LEVELS.
    """
    if level not in VALID_OPT_LEVELS:
        raise ValueError(
            f"passes_for_level: opt_level must be one of {VALID_OPT_LEVELS}, got {level!r}"
        )
    if level == 0:
        return []
    if level == 1:
        return [
            RemoveIdentity(isa),
            CancelInverses(isa),
            MergeRotations(isa),
        ]
    if level == 2:
        return [
            RemoveIdentity(isa),
            CancelInverses(isa),
            MergeRotations(isa),
            FuseEulerZYZ(isa),
            HCXHRule(isa),
            SwapElision(isa),
        ]
    # level == 3
    return [
        RemoveIdentity(isa),
        CancelInverses(isa),
        MergeRotations(isa),
        FuseEulerZYZ(isa),
        HCXHRule(isa),
        SwapElision(isa),
        CommuteThroughControl(isa),
    ]


def max_iters_for_level(level: int) -> int:
    """Return the fixpoint max_iters for the arch_opt group at the given level."""
    if level not in VALID_OPT_LEVELS:
        raise ValueError(
            f"max_iters_for_level: opt_level must be one of {VALID_OPT_LEVELS}, got {level!r}"
        )
    return {0: 1, 1: 3, 2: 5, 3: 10}[level]


from .peephole_mach import (
    RemoveIdentityMach,
    MergeAdjacent1Q,
    EulerXYRecompose,
)
from .scheduling_mach import (
    DeferMeasurement,
    ParallelizeLayers,
)
from .native_synthesis import RyRzRyToHardware


# Name -> Pass class registry for opt_passes resolution at mach level.
# Every value must be constructible as cls(isa).
ALL_MACH_PASSES: dict[str, Type[Pass]] = {
    RemoveIdentityMach.name: RemoveIdentityMach,
    MergeAdjacent1Q.name:    MergeAdjacent1Q,
    EulerXYRecompose.name:   EulerXYRecompose,
    DeferMeasurement.name:   DeferMeasurement,
    ParallelizeLayers.name:  ParallelizeLayers,
    RyRzRyToHardware.name:   RyRzRyToHardware,
}


def mach_passes_for_level(level: int, isa: ISA) -> List[Pass]:
    """Return the list of mach-level passes for the given opt_level.

    :raises ValueError: if level is not in VALID_OPT_LEVELS.
    """
    if level not in VALID_OPT_LEVELS:
        raise ValueError(
            f"mach_passes_for_level: opt_level must be one of {VALID_OPT_LEVELS}, got {level!r}"
        )
    if level == 0:
        return []
    if level == 1:
        # Safe & cheap: dedup + zero-rotation removal only.
        return [
            MergeAdjacent1Q(isa),
            RemoveIdentityMach(isa),
        ]
    if level == 2:
        # + measurable wins from the canonical rz-band collapse
        # + measurement deferral.
        return [
            MergeAdjacent1Q(isa),
            RemoveIdentityMach(isa),
            RyRzRyToHardware(isa),
            DeferMeasurement(isa),
        ]
    # level == 3
    # + aggressive single-qubit recomposition + analysis-only layer
    # tagging.
    return [
        MergeAdjacent1Q(isa),
        RemoveIdentityMach(isa),
        RyRzRyToHardware(isa),
        DeferMeasurement(isa),
        EulerXYRecompose(isa),
        ParallelizeLayers(isa),
    ]


def resolve_opt_passes(
    names: list[str],
    isa: ISA,
) -> tuple[List[Pass], List[Pass]]:
    """Resolve a list of pass names into (arch_passes, mach_passes).

    Each name is looked up in ALL_ARCH_PASSES first; if absent, in
    ALL_MACH_PASSES. Unknown names raise ValueError. Order within
    each output list matches the order in *names* (filtered to that
    level).

    :raises TypeError: if names is not a list[str].
    :raises ValueError: if any name is unknown to both registries.
    """
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise TypeError("resolve_opt_passes: names must be list[str]")
    arch_out: List[Pass] = []
    mach_out: List[Pass] = []
    for name in names:
        cls = ALL_ARCH_PASSES.get(name)
        if cls is not None:
            arch_out.append(cls(isa))
            continue
        cls = ALL_MACH_PASSES.get(name)
        if cls is not None:
            mach_out.append(cls(isa))
            continue
        raise ValueError(f"Unknown pass: {name}")
    return arch_out, mach_out
