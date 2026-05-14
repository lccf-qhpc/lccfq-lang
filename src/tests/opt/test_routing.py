"""
Filename: test_routing.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.0
Description:
    Unit tests for LookaheadSwapInsertion and LayoutSelection (Phase 4).

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import pytest
from lccfq_lang.arch.instruction import Instruction, InstructionType
from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.mapping import QPUMapping
from lccfq_lang.mach.topology import QPUTopology
from lccfq_lang.opt.builtin.routing import (
    LookaheadSwapInsertion,
    LayoutSelection,
    _two_qubit_qubits,
    _all_pairs_distance,
    _bfs_distance,
    _score_swap,
    _rewrite,
)
from lccfq_lang.opt.pass_base import PassContext
from lccfq_lang.sys.base import QPUConfig
from tests.opt._equiv import assert_equivalent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isa():
    return ISA("test")


@pytest.fixture
def linear4_spec():
    return {
        "qpu": {
            "name": "pfaff_v1",
            "location": "lab",
            "topology": "linear",
            "qubit_count": 4,
            "qubits": [0, 1, 2, 3],
            "couplings": [(0, 1), (1, 2), (2, 3)],
            "exclusions": [],
        },
        "network": {"ip": "127.0.0.1", "port": 1234},
    }


@pytest.fixture
def linear3_spec():
    return {
        "qpu": {
            "name": "pfaff_v1",
            "location": "lab",
            "topology": "linear",
            "qubit_count": 3,
            "qubits": [0, 1, 2],
            "couplings": [(0, 1), (1, 2)],
            "exclusions": [],
        },
        "network": {"ip": "127.0.0.1", "port": 1234},
    }


@pytest.fixture
def topo4(linear4_spec):
    return QPUTopology(QPUConfig(linear4_spec))


@pytest.fixture
def topo3(linear3_spec):
    return QPUTopology(QPUConfig(linear3_spec))


@pytest.fixture
def ctx4(topo4, isa):
    return PassContext(topology=topo4, isa=isa)


@pytest.fixture
def ctx3(topo3, isa):
    return PassContext(topology=topo3, isa=isa)


def _make_mapped(instr: Instruction) -> Instruction:
    """Deep-copy an instruction and mark it is_mapped=True."""
    out = Instruction(
        symbol=instr.symbol,
        modifies_state=instr.modifies_state,
        is_controlled=instr.is_controlled,
        target_qubits=list(instr.target_qubits) if instr.target_qubits else [],
        control_qubits=list(instr.control_qubits) if instr.control_qubits else [],
        params=instr.params,
        shots=instr.shots,
    )
    out.instruction_type = InstructionType.DELAYED
    out.pre = instr.pre.copy()
    out.post = instr.post.copy()
    out.is_mapped = True
    return out


# ---------------------------------------------------------------------------
# Test 1: passthrough when already adjacent
# ---------------------------------------------------------------------------

def test_passthrough_when_already_adjacent(isa, topo4, ctx4):
    """A program of 2q gates on adjacent qubits is emitted with zero SWAPs,
    same length, and same gate symbol order."""
    # cx(0,1) and cx(1,2) are directly adjacent on 0-1-2-3 linear.
    program = [
        _make_mapped(isa.cx(ct=0, tg=1)),
        _make_mapped(isa.cx(ct=1, tg=2)),
        _make_mapped(isa.cx(ct=2, tg=3)),
    ]
    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    result = pass_inst.run(program, ctx4)

    swaps = [i for i in result if i.symbol == "swap"]
    assert len(swaps) == 0, "No SWAPs expected for adjacent gates"
    assert len(result) == len(program)
    for orig, out in zip(program, result):
        assert out.symbol == orig.symbol
    # All outputs must be marked is_mapped.
    assert all(i.is_mapped for i in result)


# ---------------------------------------------------------------------------
# Test 2: inserts SWAP for a non-adjacent gate
# ---------------------------------------------------------------------------

def test_inserts_swap_for_non_adjacent(isa, topo4, ctx4):
    """CNOT(0, 3) on linear[0..3] must have at least one SWAP inserted.
    After routing, the CNOT gate is on an adjacent edge."""
    program = [_make_mapped(isa.cx(ct=0, tg=3))]
    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    result = pass_inst.run(program, ctx4)

    swaps = [i for i in result if i.symbol == "swap"]
    assert len(swaps) >= 1, "Expected at least one SWAP for cx(0,3)"

    # The CNOT in the output must be on an adjacent edge.
    cxs = [i for i in result if i.symbol == "cx"]
    assert len(cxs) == 1
    cx_out = cxs[0]
    q0 = cx_out.control_qubits[0]
    q1 = cx_out.target_qubits[0]
    assert topo4.internal.has_edge(q0, q1), (
        f"Routed cx({q0},{q1}) is not on a topology edge"
    )
    assert all(i.is_mapped for i in result)


# ---------------------------------------------------------------------------
# Test 3: lookahead picks a better path than per-instruction greedy
# ---------------------------------------------------------------------------

def test_lookahead_picks_better_path(isa, topo4, ctx4):
    """cx(0,3); cx(3,0); cx(0,3) on linear[0..3].

    SABRE-lite SWAP count must be strictly less than the per-instruction
    greedy count from QPUTopology.swaps().
    """
    # Build already-mapped program.
    raw = [isa.cx(ct=0, tg=3), isa.cx(ct=3, tg=0), isa.cx(ct=0, tg=3)]
    program = [_make_mapped(i) for i in raw]

    # SABRE-lite count.
    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    routed = pass_inst.run(list(program), ctx4)
    sabre_swaps = sum(1 for i in routed if i.symbol == "swap")

    # Greedy count: sum QPUTopology.swaps per instruction.
    greedy_swaps = sum(
        sum(1 for g in topo4.swaps(instr, isa) if g.symbol == "swap")
        for instr in program
    )

    assert sabre_swaps < greedy_swaps, (
        f"SABRE-lite ({sabre_swaps}) should be strictly less than greedy ({greedy_swaps})"
    )
    assert sabre_swaps <= 4, (
        f"Expected SABRE-lite SWAP count <= 4, got {sabre_swaps}"
    )


# ---------------------------------------------------------------------------
# Test 4: LayoutSelection improves SWAP count
# ---------------------------------------------------------------------------

def test_layout_improves_swap_count(isa, topo4):
    """Repeated cx(0,3)/cx(3,0) fixture: LayoutSelection must find a layout
    strictly better than zip-order (fewer SWAPs under SABRE-lite)."""
    # Virtual program (unmapped qubits 0..3).
    program = []
    for _ in range(5):
        program.append(isa.cx(ct=0, tg=3))
        program.append(isa.cx(ct=3, tg=0))

    initial_layout = {0: 0, 1: 1, 2: 2, 3: 3}
    baseline = LayoutSelection._count_swaps(program, initial_layout, topo4, isa)

    best_layout = LayoutSelection.compute_layout(
        program, topo4, isa, initial_layout
    )
    best_count = LayoutSelection._count_swaps(program, best_layout, topo4, isa)

    assert best_count < baseline, (
        f"LayoutSelection should improve SWAP count: {baseline} -> {best_count}"
    )


# ---------------------------------------------------------------------------
# Test 5: routing_strategy validation
# ---------------------------------------------------------------------------

def test_routing_strategy_validation(topo4):
    """QPUMapping with an invalid routing_strategy must raise ValueError with
    the exact error message specified in §14."""
    with pytest.raises(ValueError) as exc_info:
        QPUMapping([0, 1], topo4, routing_strategy="bogus")

    assert str(exc_info.value) == (
        "QPUMapping: routing_strategy must be one of "
        "('identity', 'sabre_lite'), got 'bogus'"
    )


# ---------------------------------------------------------------------------
# Test 6: with_layout is non-destructive
# ---------------------------------------------------------------------------

def test_with_layout_non_destructive(topo4):
    """m.with_layout({...}) must NOT mutate m.mapping."""
    m = QPUMapping([0, 1], topo4)
    original_mapping = dict(m.mapping)  # {0: 0, 1: 1}

    m2 = m.with_layout({0: 1, 1: 0})

    assert m.mapping == original_mapping, (
        "Original QPUMapping.mapping was mutated by with_layout"
    )
    assert m2.mapping == {0: 1, 1: 0}
    assert m2.routing_strategy == m.routing_strategy
    assert m2.topology is m.topology


# ---------------------------------------------------------------------------
# Test 7: with_layout validates keys
# ---------------------------------------------------------------------------

def test_with_layout_validates_keys(topo4):
    """Supplying mismatched virtual qubit keys raises ValueError."""
    m = QPUMapping([0, 1], topo4)

    with pytest.raises(ValueError, match="new_layout keys must equal virtual_qubits"):
        m.with_layout({0: 0, 2: 1})  # key 2 is not in virtual_qubits


def test_with_layout_validates_unique_physical(topo4):
    """Duplicate physical assignments raise ValueError."""
    m = QPUMapping([0, 1], topo4)
    with pytest.raises(ValueError, match="physical assignments must be unique"):
        m.with_layout({0: 0, 1: 0})  # both mapped to physical 0


def test_with_layout_validates_physical_in_topology(topo4):
    """Physical values not in topology raise ValueError."""
    m = QPUMapping([0, 1], topo4)
    with pytest.raises(ValueError, match="physical values must be topology qubits"):
        m.with_layout({0: 0, 1: 99})  # 99 is not in the 4-qubit topology


# ---------------------------------------------------------------------------
# Test 8: routing is deterministic
# ---------------------------------------------------------------------------

def test_routing_is_deterministic(isa, topo4, ctx4):
    """Running LookaheadSwapInsertion twice on the same input must produce
    byte-identical output (compared by repr)."""
    raw = [
        isa.cx(ct=0, tg=3),
        isa.cx(ct=3, tg=0),
        isa.cx(ct=0, tg=3),
    ]
    program = [_make_mapped(i) for i in raw]

    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    out1 = pass_inst.run(list(program), ctx4)
    out2 = pass_inst.run(list(program), ctx4)

    assert repr(out1) == repr(out2), (
        "LookaheadSwapInsertion produced non-deterministic output"
    )


# ---------------------------------------------------------------------------
# Test 9: unmappable gate raises RuntimeError
# ---------------------------------------------------------------------------

def test_unmappable_raises(isa, topo4):
    """A 2q gate whose qubits can never become adjacent must raise RuntimeError.

    We simulate this by removing all edges from the topology graph after
    construction, making routing impossible.
    """
    # Build a valid 4-qubit linear topology, then disconnect it.
    program = [_make_mapped(isa.cx(ct=0, tg=3))]

    # Disconnect the topology by removing all edges.
    edges = list(topo4.internal.edges)
    for u, v in edges:
        topo4.internal.remove_edge(u, v)

    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    ctx = PassContext(topology=topo4, isa=isa)

    with pytest.raises(RuntimeError, match="LookaheadSwapInsertion: program contains an unmappable 2q gate"):
        pass_inst.run(program, ctx)


# ---------------------------------------------------------------------------
# Test 10: equivalence on 1q-only program (passthrough check)
# ---------------------------------------------------------------------------

def test_equivalence_1q_only(isa, topo4, ctx4):
    """1q-only program run through LookaheadSwapInsertion should produce
    output equivalent to the input (no SWAPs, same state vector)."""
    # Physical 1q program (already mapped to physical qubits 0,1,2,3).
    raw_program = [
        isa.h(tg=0),
        isa.x(tg=1),
        isa.h(tg=2),
        isa.x(tg=0),
    ]
    program = [_make_mapped(i) for i in raw_program]

    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    result = pass_inst.run(list(program), ctx4)

    # No SWAPs expected for 1q program.
    assert not any(i.symbol == "swap" for i in result)
    assert len(result) == len(program)

    # Semantic equivalence: same statevector on 4 qubits.
    assert_equivalent(program, result, n_qubits=4)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_two_qubit_qubits_returns_none_for_measure(isa):
    instr = isa.measure(tgs=[0, 1])
    assert _two_qubit_qubits(instr) is None


def test_two_qubit_qubits_returns_none_for_reset(isa):
    instr = isa.reset(tgs=[0])
    assert _two_qubit_qubits(instr) is None


def test_two_qubit_qubits_returns_none_for_1q(isa):
    instr = isa.h(tg=0)
    assert _two_qubit_qubits(instr) is None


def test_two_qubit_qubits_returns_pair_for_cx(isa):
    instr = isa.cx(ct=0, tg=1)
    pair = _two_qubit_qubits(instr)
    assert pair == (0, 1)


def test_all_pairs_distance_correctness(topo4):
    """BFS distances on 0-1-2-3 linear topology are correct."""
    dist = _all_pairs_distance(topo4)
    assert dist[(0, 3)] == 3
    assert dist[(0, 1)] == 1
    assert dist[(1, 3)] == 2
    assert dist[(0, 0)] == 0


def test_bfs_distance_correctness(topo4):
    assert _bfs_distance(topo4, 0, 3) == 3
    assert _bfs_distance(topo4, 0, 0) == 0
    assert _bfs_distance(topo4, 1, 3) == 2


def test_rewrite_preserves_attributes(isa):
    """_rewrite must copy all attributes and remap qubit indices."""
    instr = isa.cx(ct=0, tg=1)
    instr.is_mapped = True
    layout = {0: 2, 1: 3}
    out = _rewrite(instr, layout)

    assert out.symbol == "cx"
    assert out.control_qubits == [2]
    assert out.target_qubits == [3]
    assert out.is_mapped is True
    assert out is not instr  # must be a new object


def test_empty_program_returns_empty(isa, topo4, ctx4):
    """Running LookaheadSwapInsertion on an empty program returns []."""
    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    result = pass_inst.run([], ctx4)
    assert result == []


def test_1q_only_no_swaps_mixed_with_measure(isa, topo4, ctx4):
    """1q gates and measure instructions pass through with no SWAPs."""
    program = [
        _make_mapped(isa.h(tg=0)),
        _make_mapped(isa.x(tg=1)),
        _make_mapped(isa.measure(tgs=[0, 1])),
    ]
    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    result = pass_inst.run(program, ctx4)

    assert not any(i.symbol == "swap" for i in result)
    assert len(result) == 3
