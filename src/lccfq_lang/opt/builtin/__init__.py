"""
Filename: __init__.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Public API for the lccfq_lang.opt.builtin sub-package.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from .lower_passes import (
    MappedPass,
    SwappedPass,
    TranspiledPass,
    build_lowering_groups,
    slice_groups_for,
    LOWERING_STAGES,
)
from .lower_universal import LowerU2, LowerU3, LowerCU, FanoutMeasure
from .peephole_arch import (
    RemoveIdentity,
    CancelInverses,
    MergeRotations,
    FuseEulerZYZ,
    CommuteThroughControl,
)
from .templates_arch import (
    HCXHRule,
    SwapElision,
    TEMPLATE_REGISTRY,
    register_template,
    unregister_template,
    get_registered_templates,
)
from .level_select import (
    passes_for_level,
    mach_passes_for_level,
    max_iters_for_level,
    resolve_opt_passes,
    ALL_ARCH_PASSES,
    ALL_MACH_PASSES,
    VALID_OPT_LEVELS,
)
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
from ._native import (
    NATIVE_1Q_PARAM,
    NATIVE_2Q,
    NATIVE_MEASURE,
    NATIVE_RESET,
    NATIVE_ALL_SYMBOLS,
)
from .routing import LookaheadSwapInsertion, LayoutSelection

__all__ = [
    "MappedPass",
    "SwappedPass",
    "TranspiledPass",
    "build_lowering_groups",
    "slice_groups_for",
    "LOWERING_STAGES",
    "LowerU2",
    "LowerU3",
    "LowerCU",
    "FanoutMeasure",
    "RemoveIdentity",
    "CancelInverses",
    "MergeRotations",
    "FuseEulerZYZ",
    "CommuteThroughControl",
    "HCXHRule",
    "SwapElision",
    "TEMPLATE_REGISTRY",
    "register_template",
    "unregister_template",
    "get_registered_templates",
    "passes_for_level",
    "mach_passes_for_level",
    "max_iters_for_level",
    "resolve_opt_passes",
    "ALL_ARCH_PASSES",
    "ALL_MACH_PASSES",
    "VALID_OPT_LEVELS",
    "RemoveIdentityMach",
    "MergeAdjacent1Q",
    "EulerXYRecompose",
    "DeferMeasurement",
    "ParallelizeLayers",
    "RyRzRyToHardware",
    "NATIVE_1Q_PARAM",
    "NATIVE_2Q",
    "NATIVE_MEASURE",
    "NATIVE_RESET",
    "NATIVE_ALL_SYMBOLS",
    "LookaheadSwapInsertion",
    "LayoutSelection",
]
