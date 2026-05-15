"""
Filename: __init__.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.1
Description:
    This file selectively exposes a curated interface for user-level programming with
    lccfq_lang. Phase 5 adds re-exports for the optimization API so custom-pass
    authors can write `from lccfq_lang import Pass, PassManager, ...` without
    reaching into `lccfq_lang.opt.*`.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from .backend import QPU
from .arch.register import QRegister, CRegister
from .arch.context import Circuit, Test
from .arch.isa import ISA
from .arch.synth.qasm import QASMSynthesizer

# Phase 5: optimization API surface ------------------------------------------
from .opt.pass_base import Pass, PassContext, PassRecord
from .opt.manager   import PassGroup, PassManager
from .opt.cost      import Cost
from .opt.op_view   import OpView
from .opt.dag       import circuit_to_dag, dag_to_program
from .opt.builtin.templates_arch import (
    register_template,
    unregister_template,
    get_registered_templates,
    TEMPLATE_REGISTRY,
)
from .opt.builtin.level_select import (
    ALL_ARCH_PASSES,
    ALL_MACH_PASSES,
    passes_for_level,
    mach_passes_for_level,
    VALID_OPT_LEVELS,
)


__all__ = [
    # Existing exports (Phase 0/1/2/3/4)
    "QPU",
    "QRegister",
    "CRegister",
    "Circuit",
    "ISA",
    "QASMSynthesizer",
    # Phase 5 additions
    "Pass",
    "PassContext",
    "PassRecord",
    "PassGroup",
    "PassManager",
    "Cost",
    "OpView",
    "circuit_to_dag",
    "dag_to_program",
    "register_template",
    "unregister_template",
    "get_registered_templates",
    "TEMPLATE_REGISTRY",
    "ALL_ARCH_PASSES",
    "ALL_MACH_PASSES",
    "passes_for_level",
    "mach_passes_for_level",
    "VALID_OPT_LEVELS",
]
