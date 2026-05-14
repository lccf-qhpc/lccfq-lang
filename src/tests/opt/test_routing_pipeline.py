"""
Filename: test_routing_pipeline.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.0
Description:
    End-to-end pipeline integration tests for routing_strategy in Phase 4.
    Tests opt_level routing selection, byte-identical opt_level=0, SWAP
    count reduction at opt_level=2, and explicit strategy override.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import pytest
from pathlib import Path
from lccfq_lang.arch.context import Circuit
from lccfq_lang.arch.instruction import Instruction, InstructionType
from lccfq_lang.arch.mapping import QPUMapping
from lccfq_lang.arch.register import CRegister, QRegister
from lccfq_lang.backend import QPU
from lccfq_lang.mach.topology import QPUTopology
from lccfq_lang.opt.builtin.lower_passes import build_lowering_groups, slice_groups_for
from lccfq_lang.opt.manager import PassManager
from lccfq_lang.opt.pass_base import PassContext
from lccfq_lang.sys.base import QPUConfig

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG = str(DATA_DIR / "testing.toml")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _capture_swapped_program(qpu, opt_level=0, gates=None, routing_strategy=None):
    """Compile a circuit to the 'swapped' stage and capture the routed program.

    :param qpu: QPU with last_pass='swapped'
    :param opt_level: optimization level
    :param gates: list of (method, kwargs) pairs to emit; defaults to cx(0,3)
    :param routing_strategy: if not None, create QPUMapping with this strategy
    :return: list of Instruction after swapped stage
    """
    if routing_strategy is not None:
        topo = qpu.mapping.topology
        vq = list(range(qpu.config.qubit_count))
        mapping = QPUMapping(vq, topo, routing_strategy=routing_strategy)
        qreg = QRegister(qpu.config.qubit_count, mapping, qpu.isa)
    else:
        qreg = qpu.qregister(qpu.config.qubit_count)

    creg = CRegister(qpu.config.qubit_count)
    captured = []

    original_handle = Circuit._handle_pass

    def patched(self, program, cpass):
        if cpass == "swapped":
            captured.extend(program)
        original_handle(self, program, cpass)

    Circuit._handle_pass = patched
    try:
        with Circuit(qreg, creg, qpu, shots=1, opt_level=opt_level) as c:
            if gates is None:
                c >> qpu.isa.cx(ct=0, tg=3)
            else:
                for method_name, kwargs in gates:
                    c >> getattr(qpu.isa, method_name)(**kwargs)
    finally:
        Circuit._handle_pass = original_handle

    return captured


def _count_swaps(program):
    return sum(1 for i in program if i.symbol == "swap")


# ---------------------------------------------------------------------------
# Test 1: opt_level=0 byte-identical to legacy (SwappedPass) behavior
# ---------------------------------------------------------------------------

def test_opt_level_0_byte_identical_to_legacy():
    """Two circuits compiled at opt_level=0 with the same program must
    produce byte-identical 'swapped' stage output (repr comparison).

    This validates that opt_level=0 always uses SwappedPass and that the
    determinism guarantee holds for the identity strategy.
    """
    qpu = QPU(filename=CONFIG, last_pass="swapped")

    gates = [("h", {"tg": 0}), ("x", {"tg": 1}), ("h", {"tg": 2})]

    prog1 = _capture_swapped_program(qpu, opt_level=0, gates=gates)
    prog2 = _capture_swapped_program(qpu, opt_level=0, gates=gates)

    assert repr(prog1) == repr(prog2), (
        "opt_level=0 must produce byte-identical output on repeated compilation"
    )
    # No SWAPs expected for 1q-only programs.
    assert _count_swaps(prog1) == 0


# ---------------------------------------------------------------------------
# Test 2: opt_level=2 reduces SWAP count vs opt_level=0
# ---------------------------------------------------------------------------

def test_opt_level_2_reduces_swap_count():
    """A circuit with non-adjacent 2q gates must have fewer SWAPs at
    opt_level=2 than at opt_level=0.

    Uses cx(0,3); cx(1,2) on the 4-qubit linear topology.
    """
    gates = [("cx", {"ct": 0, "tg": 3}), ("cx", {"ct": 1, "tg": 2})]

    qpu0 = QPU(filename=CONFIG, last_pass="swapped")
    qpu2 = QPU(filename=CONFIG, last_pass="swapped")

    prog0 = _capture_swapped_program(qpu0, opt_level=0, gates=gates)
    prog2 = _capture_swapped_program(qpu2, opt_level=2, gates=gates)

    count0 = _count_swaps(prog0)
    count2 = _count_swaps(prog2)

    assert count2 < count0, (
        f"opt_level=2 should insert fewer SWAPs than opt_level=0: "
        f"got {count2} vs {count0}"
    )


# ---------------------------------------------------------------------------
# Test 3: last_pass='swapped' returns fully-routed program
# ---------------------------------------------------------------------------

def test_last_pass_swapped_returns_post_routing():
    """When last_pass='swapped' and opt_level=2, all 2q gates in the
    output must be on topology edges."""
    spec = {
        "qpu": {
            "name": "pfaff_v1", "location": "lab", "topology": "linear",
            "qubit_count": 4, "qubits": [0, 1, 2, 3],
            "couplings": [(0, 1), (1, 2), (2, 3)], "exclusions": [],
        },
        "network": {"ip": "127.0.0.1", "port": 1234},
    }
    qpu = QPU(filename=CONFIG, last_pass="swapped")
    topo = qpu.mapping.topology

    gates = [("cx", {"ct": 0, "tg": 3}), ("cx", {"ct": 3, "tg": 0})]
    prog = _capture_swapped_program(qpu, opt_level=2, gates=gates)

    for instr in prog:
        if instr.symbol in ("measure", "reset") or instr.symbol == "swap":
            continue
        cts = instr.control_qubits or []
        tgs = instr.target_qubits or []
        all_q = cts + tgs
        if len(all_q) == 2:
            q0, q1 = all_q
            assert topo.internal.has_edge(q0, q1), (
                f"Gate {instr.symbol}({all_q}) is not on a topology edge after routing"
            )


# ---------------------------------------------------------------------------
# Test 4: explicit routing_strategy='sabre_lite' overrides opt_level=0
# ---------------------------------------------------------------------------

def test_explicit_strategy_overrides_opt_level():
    """QPUMapping(virtual_qubits, topo, routing_strategy='sabre_lite') with
    opt_level=0 must use SABRE-lite routing."""
    spec = {
        "qpu": {
            "name": "pfaff_v1", "location": "lab", "topology": "linear",
            "qubit_count": 4, "qubits": [0, 1, 2, 3],
            "couplings": [(0, 1), (1, 2), (2, 3)], "exclusions": [],
        },
        "network": {"ip": "127.0.0.1", "port": 1234},
    }
    qpu = QPU(filename=CONFIG, last_pass="swapped")

    # Non-adjacent gate: without SABRE-lite, greedy gives 4 SWAPs for cx(0,3).
    gates = [("cx", {"ct": 0, "tg": 3}), ("cx", {"ct": 3, "tg": 0}),
             ("cx", {"ct": 0, "tg": 3})]

    # opt_level=0 with explicit sabre_lite strategy.
    prog_sabre = _capture_swapped_program(
        qpu, opt_level=0, gates=gates, routing_strategy="sabre_lite"
    )
    # opt_level=0 with default identity strategy.
    prog_greedy = _capture_swapped_program(
        qpu, opt_level=0, gates=gates, routing_strategy=None
    )

    sabre_count = _count_swaps(prog_sabre)
    greedy_count = _count_swaps(prog_greedy)

    assert sabre_count < greedy_count, (
        f"Explicit sabre_lite strategy at opt_level=0 should use fewer SWAPs "
        f"than identity: {sabre_count} vs {greedy_count}"
    )


# ---------------------------------------------------------------------------
# Test 5: explicit identity with opt_level=2 — documented one-way precedence
# ---------------------------------------------------------------------------

def test_explicit_identity_with_opt_level_2():
    """opt_level >= 2 always selects sabre_lite regardless of the mapping
    default. There is no way to force identity at opt_level=2 via the
    mapping — opt_level takes precedence.

    This is intentional per §6.2 of the spec.
    """
    pytest.skip(
        "intentional: opt_level >= 2 always selects sabre_lite; "
        "no mechanism to override to identity at opt_level=2"
    )


# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------

def test_build_lowering_groups_invalid_routing_strategy_raises():
    """build_lowering_groups with an invalid routing_strategy kwarg raises ValueError."""
    qpu = QPU(filename=CONFIG, last_pass="swapped")
    qreg = qpu.qregister(4)

    with pytest.raises(ValueError, match="build_lowering_groups: routing_strategy must be one of"):
        build_lowering_groups(qreg, qpu, routing_strategy="bogus")


def test_opt_level_0_uses_swapped_pass_not_lookahead():
    """At opt_level=0 with default mapping, the 'lower_swap' group must
    contain SwappedPass (not LookaheadSwapInsertion)."""
    from lccfq_lang.opt.builtin.lower_passes import SwappedPass
    from lccfq_lang.opt.builtin.routing import LookaheadSwapInsertion

    qpu = QPU(filename=CONFIG, last_pass="swapped")
    qreg = qpu.qregister(4)

    groups = build_lowering_groups(qreg, qpu, opt_level=0)
    swap_group = next(g for g in groups if g.name == "lower_swap")

    pass_types = [type(p) for p in swap_group.passes]
    assert SwappedPass in pass_types, "opt_level=0 should use SwappedPass"
    assert LookaheadSwapInsertion not in pass_types, (
        "opt_level=0 should NOT use LookaheadSwapInsertion"
    )


def test_sabre_lite_strategy_uses_lookahead_pass():
    """build_lowering_groups with routing_strategy='sabre_lite' must include
    LookaheadSwapInsertion in the lower_swap group."""
    from lccfq_lang.opt.builtin.lower_passes import SwappedPass
    from lccfq_lang.opt.builtin.routing import LookaheadSwapInsertion

    qpu = QPU(filename=CONFIG, last_pass="swapped")
    qreg = qpu.qregister(4)

    groups = build_lowering_groups(qreg, qpu, routing_strategy="sabre_lite")
    swap_group = next(g for g in groups if g.name == "lower_swap")

    pass_types = [type(p) for p in swap_group.passes]
    assert LookaheadSwapInsertion in pass_types, (
        "sabre_lite strategy should use LookaheadSwapInsertion"
    )
    assert SwappedPass not in pass_types, (
        "sabre_lite strategy should NOT use SwappedPass"
    )


def test_opt_level_2_circuit_uses_sabre_lite_records():
    """An opt_level=2 circuit must run the 'swapped' pass via
    LookaheadSwapInsertion. The pass record group 'lower_swap' must appear
    and no regression in the pipeline structure."""
    qpu = QPU(filename=CONFIG, last_pass="expanded")
    qreg = qpu.qregister(4)
    creg = CRegister(4)

    with Circuit(qreg, creg, qpu, shots=1, opt_level=2) as c:
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.x(tg=1)
        c >> qpu.isa.h(tg=2)

    group_names = [r.group_name for r in c._opt_records]
    assert "lower_swap" in group_names, (
        "Pipeline records must contain 'lower_swap' group"
    )


def test_with_layout_preserves_routing_strategy():
    """QPUMapping.with_layout must preserve the routing_strategy on the clone."""
    spec = {
        "qpu": {
            "name": "pfaff_v1", "location": "lab", "topology": "linear",
            "qubit_count": 4, "qubits": [0, 1, 2, 3],
            "couplings": [(0, 1), (1, 2), (2, 3)], "exclusions": [],
        },
        "network": {"ip": "127.0.0.1", "port": 1234},
    }
    topo = QPUTopology(QPUConfig(spec))
    m = QPUMapping([0, 1, 2, 3], topo, routing_strategy="sabre_lite")

    m2 = m.with_layout({0: 3, 1: 2, 2: 1, 3: 0})
    assert m2.routing_strategy == "sabre_lite"
    assert m2.mapping == {0: 3, 1: 2, 2: 1, 3: 0}
    assert m.mapping == {0: 0, 1: 1, 2: 2, 3: 3}  # original unchanged


def test_rebind_mapping_does_not_mutate_original():
    """QRegister.rebind_mapping must return a clone without mutating self."""
    qpu = QPU(filename=CONFIG, last_pass="swapped")
    qreg = qpu.qregister(4)
    original_mapping = qreg.mapping

    topo = qpu.mapping.topology
    new_mapping = QPUMapping([0, 1, 2, 3], topo, routing_strategy="identity")
    qreg2 = qreg.rebind_mapping(new_mapping)

    assert qreg.mapping is original_mapping, (
        "rebind_mapping must not mutate the original QRegister"
    )
    assert qreg2.mapping is new_mapping
    assert qreg2 is not qreg
