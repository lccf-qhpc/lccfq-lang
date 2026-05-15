"""
Filename: test_top_level_imports.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.0
Description:
    Verifies that every name promised by the top-level `lccfq_lang` package
    re-exports (Phase 5) is importable from the package root and refers to
    the canonical object.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import importlib

# (top-level name, fully-qualified module path) pairs
EXPECTED = [
    # Pre-Phase-5 (sanity)
    ("QPU",                         "lccfq_lang.backend"),
    ("QRegister",                   "lccfq_lang.arch.register"),
    ("CRegister",                   "lccfq_lang.arch.register"),
    ("Circuit",                     "lccfq_lang.arch.context"),
    ("ISA",                         "lccfq_lang.arch.isa"),
    ("QASMSynthesizer",             "lccfq_lang.arch.synth.qasm"),
    # Phase 5 additions
    ("Pass",                        "lccfq_lang.opt.pass_base"),
    ("PassContext",                 "lccfq_lang.opt.pass_base"),
    ("PassRecord",                  "lccfq_lang.opt.pass_base"),
    ("PassGroup",                   "lccfq_lang.opt.manager"),
    ("PassManager",                 "lccfq_lang.opt.manager"),
    ("Cost",                        "lccfq_lang.opt.cost"),
    ("OpView",                      "lccfq_lang.opt.op_view"),
    ("circuit_to_dag",              "lccfq_lang.opt.dag"),
    ("dag_to_program",              "lccfq_lang.opt.dag"),
    ("register_template",           "lccfq_lang.opt.builtin.templates_arch"),
    ("unregister_template",         "lccfq_lang.opt.builtin.templates_arch"),
    ("get_registered_templates",    "lccfq_lang.opt.builtin.templates_arch"),
    ("TEMPLATE_REGISTRY",           "lccfq_lang.opt.builtin.templates_arch"),
    ("ALL_ARCH_PASSES",             "lccfq_lang.opt.builtin.level_select"),
    ("ALL_MACH_PASSES",             "lccfq_lang.opt.builtin.level_select"),
    ("passes_for_level",            "lccfq_lang.opt.builtin.level_select"),
    ("mach_passes_for_level",       "lccfq_lang.opt.builtin.level_select"),
    ("VALID_OPT_LEVELS",            "lccfq_lang.opt.builtin.level_select"),
]


def test_all_advertised_names_are_importable():
    pkg = importlib.import_module("lccfq_lang")
    missing = [name for name, _ in EXPECTED if not hasattr(pkg, name)]
    assert not missing, f"Top-level missing: {missing}"


def test_top_level_objects_match_canonical_modules():
    pkg = importlib.import_module("lccfq_lang")
    for name, mod_path in EXPECTED:
        mod = importlib.import_module(mod_path)
        assert getattr(pkg, name) is getattr(mod, name), (
            f"lccfq_lang.{name} is not the same object as "
            f"{mod_path}.{name}"
        )


def test_all_entries_are_strings():
    """`__all__` must be list[str], not list of object references — otherwise
    `from lccfq_lang import *` and pydoc/sphinx introspection misbehave."""
    pkg = importlib.import_module("lccfq_lang")
    non_strings = [(i, type(x).__name__) for i, x in enumerate(pkg.__all__)
                   if not isinstance(x, str)]
    assert not non_strings, (
        f"__all__ contains non-string entries: {non_strings}"
    )


def test_star_import_exposes_all_names():
    """`from lccfq_lang import *` must surface every name in __all__."""
    ns: dict = {}
    exec("from lccfq_lang import *", ns)
    pkg = importlib.import_module("lccfq_lang")
    missing = [n for n in pkg.__all__ if n not in ns]
    assert not missing, f"`import *` did not expose: {missing}"
