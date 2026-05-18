"""
Filename: test_templates_arch.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Tests for HCXHRule, SwapElision, and the TEMPLATE_REGISTRY.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import pytest
from lccfq_lang.arch.isa import ISA
from lccfq_lang.opt.pass_base import Pass, PassContext
from lccfq_lang.opt.builtin.templates_arch import (
    HCXHRule,
    SwapElision,
    TEMPLATE_REGISTRY,
    register_template,
    unregister_template,
    get_registered_templates,
)
from tests.opt._equiv import assert_equivalent

isa = ISA("lccfq")
ctx = PassContext(isa=isa)


# ===========================================================================
# HCXHRule
# ===========================================================================

class TestHCXHRule:
    def setup_method(self):
        self.pass_ = HCXHRule(isa)

    def test_passthrough_empty(self):
        assert self.pass_.run([], ctx)[0] == []

    def test_canonical_match(self):
        # H(1) CX(0,1) H(1) -> CZ(0,1)
        program = [isa.h(tg=1), isa.cx(ct=0, tg=1), isa.h(tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 1
        assert result[0].symbol == "cz"
        assert result[0].control_qubits == [0]
        assert result[0].target_qubits == [1]

    def test_wrong_target_no_match(self):
        # H on the control (qubit 0), not the target (qubit 1) — no match
        program = [isa.h(tg=0), isa.cx(ct=0, tg=1), isa.h(tg=0)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 3

    def test_no_h_after_no_match(self):
        program = [isa.h(tg=1), isa.cx(ct=0, tg=1), isa.x(tg=1)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 3

    def test_does_not_mutate(self):
        program = [isa.h(tg=1), isa.cx(ct=0, tg=1), isa.h(tg=1)]
        original_ids = [id(i) for i in program]
        self.pass_.run(program, ctx)
        assert [id(i) for i in program] == original_ids

    def test_preserves_semantics(self):
        # H(1) CX(0,1) H(1) == CZ(0,1) semantically
        p_in = [isa.h(tg=1), isa.cx(ct=0, tg=1), isa.h(tg=1)]
        p_out, _ = self.pass_.run(p_in, ctx)
        assert_equivalent(p_in, p_out, 2)


# ===========================================================================
# SwapElision
# ===========================================================================

class TestSwapElision:
    def setup_method(self):
        self.pass_ = SwapElision(isa)

    def test_passthrough_empty(self):
        assert self.pass_.run([], ctx)[0] == []

    def test_canonical_elision(self):
        program = [isa.swap(tg_a=0, tg_b=1), isa.swap(tg_a=0, tg_b=1)]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_symmetric_elision(self):
        program = [isa.swap(tg_a=0, tg_b=1), isa.swap(tg_a=1, tg_b=0)]
        result, _ = self.pass_.run(program, ctx)
        assert result == []

    def test_different_qubits_no_match(self):
        program = [isa.swap(tg_a=0, tg_b=1), isa.swap(tg_a=0, tg_b=2)]
        result, _ = self.pass_.run(program, ctx)
        assert len(result) == 2

    def test_does_not_mutate(self):
        program = [isa.swap(tg_a=0, tg_b=1), isa.swap(tg_a=0, tg_b=1)]
        original_ids = [id(i) for i in program]
        self.pass_.run(program, ctx)
        assert [id(i) for i in program] == original_ids

    def test_preserves_semantics(self):
        p_in = [
            isa.swap(tg_a=0, tg_b=1),
            isa.swap(tg_a=0, tg_b=1),
            isa.cx(ct=0, tg=1),
        ]
        p_out, _ = self.pass_.run(p_in, ctx)
        assert_equivalent(p_in, p_out, 2)


# ===========================================================================
# TEMPLATE_REGISTRY
# ===========================================================================

# Helper dummy pass for registry tests
class _DummyPass(Pass):
    name = "dummy_template"
    applies_to = "arch"

    def __init__(self, isa: ISA) -> None:
        self._isa = isa

    def run(self, program, ctx):
        return list(program), False


class TestTemplateRegistry:
    def teardown_method(self):
        # Clean up any registered dummy template
        unregister_template("dummy_template")
        unregister_template("another_template")

    def test_register_lookup_unregister(self):
        pass_obj = _DummyPass(isa)
        register_template("dummy_template", pass_obj)
        templates = get_registered_templates()
        assert pass_obj in templates
        unregister_template("dummy_template")
        templates_after = get_registered_templates()
        assert pass_obj not in templates_after

    def test_register_duplicate_raises(self):
        pass_obj = _DummyPass(isa)
        register_template("dummy_template", pass_obj)
        with pytest.raises(ValueError, match="already registered"):
            register_template("dummy_template", _DummyPass(isa))

    def test_register_wrong_applies_to_raises(self):
        class _MachPass(Pass):
            name = "mach_pass"
            applies_to = "mach"

            def run(self, program, ctx):
                return list(program), False

        with pytest.raises(ValueError, match="applies_to.*must be.*arch"):
            register_template("mach_pass", _MachPass())

    def test_register_name_mismatch_raises(self):
        pass_obj = _DummyPass(isa)
        with pytest.raises(ValueError, match="does not match"):
            register_template("wrong_name", pass_obj)

    def test_register_non_pass_raises(self):
        with pytest.raises(TypeError, match="must be a Pass"):
            register_template("dummy_template", object())

    def test_register_non_string_name_raises(self):
        pass_obj = _DummyPass(isa)
        with pytest.raises(TypeError, match="non-empty string"):
            register_template(42, pass_obj)

    def test_register_empty_string_name_raises(self):
        pass_obj = _DummyPass(isa)
        with pytest.raises(TypeError, match="non-empty string"):
            register_template("", pass_obj)
