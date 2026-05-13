"""
Filename: test_equivalence_corpus.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Property-style randomized equivalence test for arch-level optimization
    passes. Deterministic via per-test seeds.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import random
import math

import pytest
import numpy as np

from lccfq_lang.arch.isa import ISA
from tests.opt._equiv import assert_equivalent

# Symbols supported by tests/_sim.py — restrict the corpus to these.
_SUPPORTED_SQ_NOPAR = ["h", "x", "z"]
_SUPPORTED_SQ_PAR = ["rx", "ry", "rz", "p"]
_SUPPORTED_2Q = ["cx", "cz"]
# swap is also supported by _sim.py
_SUPPORTED_2Q_SYM = ["swap"]


@pytest.mark.parametrize("seed", list(range(20)))
@pytest.mark.parametrize("opt_level", [1, 2, 3])
def test_random_circuit_equivalence(seed, opt_level):
    rng = random.Random(seed)
    isa = ISA("lccfq")
    n_qubits = rng.randint(2, 4)
    n_ops = rng.randint(5, 15)
    program = []
    for _ in range(n_ops):
        kind = rng.choice(["sqn", "sqp", "tq"])
        if kind == "sqn":
            sym = rng.choice(_SUPPORTED_SQ_NOPAR)
            q = rng.randrange(n_qubits)
            program.append(getattr(isa, sym)(tg=q))
        elif kind == "sqp":
            sym = rng.choice(_SUPPORTED_SQ_PAR)
            q = rng.randrange(n_qubits)
            theta = rng.uniform(-2 * math.pi, 2 * math.pi)
            program.append(getattr(isa, sym)(tg=q, params=[theta]))
        else:
            sym = rng.choice(_SUPPORTED_2Q)
            a, b = rng.sample(range(n_qubits), 2)
            program.append(getattr(isa, sym)(ct=a, tg=b))

    # Run arch_opt directly (without involving Circuit/QPU plumbing).
    from lccfq_lang.opt.builtin.level_select import (
        passes_for_level,
        max_iters_for_level,
    )
    from lccfq_lang.opt.manager import PassGroup, PassManager
    from lccfq_lang.opt.pass_base import PassContext

    passes = passes_for_level(opt_level, isa)
    if not passes:
        return  # opt_level == 0 not exercised here
    group = PassGroup(
        "arch_opt",
        "fixpoint",
        passes,
        max_iters=max_iters_for_level(opt_level),
    )
    optimized, _ = PassManager([group]).run(list(program), PassContext(isa=isa))

    assert_equivalent(program, optimized, n_qubits)
