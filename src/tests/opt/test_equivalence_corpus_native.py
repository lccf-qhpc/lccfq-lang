"""
Filename: test_equivalence_corpus_native.py
Author: Santiago Nunez-Corrales
Date: 2026-05-13
Version: 1.0
Description:
    Randomized end-to-end equivalence corpus for mach-level optimization.
    Builds random arch programs, runs them through the full pipeline at every
    opt_level, simulates the resulting mach program, and asserts all four
    results agree up to global phase.

    The corpus uses arch programs made of only gates that XYiSW can transpile
    to native mach gates (rx, ry, sqiswap, measure, nop).

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import random
import math
import pytest
from pathlib import Path
from lccfq_lang.arch.isa import ISA
from lccfq_lang.opt.builtin.lower_passes import build_lowering_groups, slice_groups_for
from lccfq_lang.opt.manager import PassManager
from lccfq_lang.opt.pass_base import PassContext
from lccfq_lang.backend import QPU
from tests.opt._equiv_native import assert_equivalent_native

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG = str(DATA_DIR / "testing.toml")

# Arch gates that safely transpile to native mach gates via XYiSW and can
# be verified with the mach simulator.  Two-qubit decompositions (cx, cy,
# cz) have been verified correct at ~1e-15 Frobenius precision in Task #16
# and are now included in the corpus.  The random program generator selects
# from both single-qubit and two-qubit no-parameter gates so that the corpus
# exercises the full transpilation path.  The corpus remains deterministic
# with fixed seeds.
_SUPPORTED_SQ_NOPAR = ["x", "y"]
_SUPPORTED_SQ_PAR = ["rx", "ry", "rz"]
_SUPPORTED_TQ_NOPAR = ["cx", "cy", "cz"]


def _random_arch_program(seed: int, n_qubits: int):
    """Generate a deterministic random arch program for the given seed.

    Includes single-qubit gates (parametric and non-parametric) as well as
    the three verified no-parameter two-qubit gates (cx, cy, cz).  Two-qubit
    gates are only emitted when n_qubits >= 2; control and target qubits are
    always distinct.
    """
    rng = random.Random(seed)
    isa = ISA("lccfq")
    n_ops = rng.randint(3, 10)
    program = []
    for _ in range(n_ops):
        choices = ["sqn", "sqp"]
        if n_qubits >= 2:
            choices.append("tqn")
        kind = rng.choice(choices)
        if kind == "sqn":
            sym = rng.choice(_SUPPORTED_SQ_NOPAR)
            q = rng.randrange(n_qubits)
            program.append(getattr(isa, sym)(tg=q))
        elif kind == "sqp":
            sym = rng.choice(_SUPPORTED_SQ_PAR)
            q = rng.randrange(n_qubits)
            theta = rng.uniform(-math.pi, math.pi)
            program.append(getattr(isa, sym)(tg=q, params=[theta]))
        else:  # tqn
            sym = rng.choice(_SUPPORTED_TQ_NOPAR)
            ct = rng.randrange(n_qubits)
            tg = rng.choice([q for q in range(n_qubits) if q != ct])
            program.append(getattr(isa, sym)(ct=ct, tg=tg))
    return program


def _run_to_mach(arch_program, opt_level: int, n_qubits: int):
    """Run arch_program through the full pipeline at opt_level.

    Returns the final mach-level program (List[Gate | Control | Test]).
    """
    qpu = QPU(filename=CONFIG, last_pass="mach_optimized")
    qreg = qpu.qregister(n_qubits)
    groups = build_lowering_groups(qreg, qpu, opt_level=opt_level)
    # Always run all groups including mach_opt (if present).
    ctx = PassContext(
        qpu_config=qpu.config,
        isa=qpu.isa,
        mapping=qpu.mapping,
        topology=qpu.mapping.topology,
    )
    program, _ = PassManager(groups).run(list(arch_program), ctx)
    return program


@pytest.mark.parametrize("seed", list(range(20)))
def test_native_equivalence_across_opt_levels(seed):
    """All four opt_level values must produce semantically equivalent mach programs."""
    n_qubits = 3
    # Use a fresh ISA for program generation (independent of QPU).
    arch_program = _random_arch_program(seed, n_qubits)

    results = []
    for level in (0, 1, 2, 3):
        prog = _run_to_mach(arch_program, level, n_qubits)
        results.append(prog)

    ref = results[0]
    for prog in results[1:]:
        assert_equivalent_native(ref, prog, n_qubits)


@pytest.mark.parametrize("seed", list(range(10)))
def test_native_equivalence_single_qubit(seed):
    """Single-qubit programs must remain equivalent across opt_levels."""
    n_qubits = 1
    rng = random.Random(seed + 1000)
    isa = ISA("lccfq")
    n_ops = rng.randint(3, 8)
    program = []
    for _ in range(n_ops):
        sym = rng.choice(_SUPPORTED_SQ_PAR)
        theta = rng.uniform(-math.pi, math.pi)
        program.append(getattr(isa, sym)(tg=0, params=[theta]))

    results = []
    for level in (0, 1, 2, 3):
        prog = _run_to_mach(program, level, n_qubits)
        results.append(prog)

    ref = results[0]
    for prog in results[1:]:
        assert_equivalent_native(ref, prog, n_qubits)


@pytest.mark.parametrize("seed", list(range(10)))
def test_native_equivalence_multi_qubit(seed):
    """Multi-qubit (single-qubit gates only) programs must be equivalent across levels."""
    n_qubits = 2
    arch_program = _random_arch_program(seed + 2000, n_qubits)
    results = []
    for level in (0, 1, 2, 3):
        prog = _run_to_mach(arch_program, level, n_qubits)
        results.append(prog)
    ref = results[0]
    for prog in results[1:]:
        assert_equivalent_native(ref, prog, n_qubits)


def test_rz_band_collapse_equivalence():
    """Two consecutive rz ops at opt_level>=2 should produce equivalent output."""
    isa = ISA("lccfq")
    program = [
        isa.rz(tg=0, params=[0.7]),
        isa.rz(tg=0, params=[0.4]),
    ]
    prog_level0 = _run_to_mach(program, 0, n_qubits=1)
    prog_level2 = _run_to_mach(program, 2, n_qubits=1)

    # At level=2, the 6-gate band collapses to 3 gates.
    assert len(prog_level0) == 6  # two full rz bands
    assert len(prog_level2) == 3  # one collapsed band

    assert_equivalent_native(prog_level0, prog_level2, n_qubits=1)


def test_adjacent_merge_equivalence():
    """Two rx ops on the same qubit should merge at opt_level>=1."""
    isa = ISA("lccfq")
    program = [
        isa.rx(tg=0, params=[0.5]),
        isa.rx(tg=0, params=[0.3]),
    ]
    prog_level0 = _run_to_mach(program, 0, n_qubits=1)
    prog_level1 = _run_to_mach(program, 1, n_qubits=1)

    # At level>=1, the two rx ops merge to one.
    assert len(prog_level0) == 2
    assert len(prog_level1) == 1
    assert_equivalent_native(prog_level0, prog_level1, n_qubits=1)
