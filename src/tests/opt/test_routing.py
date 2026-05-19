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
    LayoutSelectionPass,
    _two_qubit_qubits,
    _all_pairs_distance,
    _bfs_distance,
    _score_swap,
    _rewrite,
    _dedup_unique_2q_pairs,
    _proxy_cost,
    _effective_max_rounds,
    MAX_ROUNDS_DEFAULT,
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
    result, _ = pass_inst.run(program, ctx4)

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
    result, _ = pass_inst.run(program, ctx4)

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
    routed, _ = pass_inst.run(list(program), ctx4)
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
    the exact error message specified in §14 (updated for Perf #11)."""
    with pytest.raises(ValueError) as exc_info:
        QPUMapping([0, 1], topo4, routing_strategy="bogus")

    assert str(exc_info.value) == (
        "QPUMapping: routing_strategy must be one of "
        "('identity', 'sabre_lite', 'sabre_fast'), got 'bogus'"
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
    out1, _ = pass_inst.run(list(program), ctx4)
    out2, _ = pass_inst.run(list(program), ctx4)

    assert repr(out1) == repr(out2), (
        "LookaheadSwapInsertion produced non-deterministic output"
    )


# ---------------------------------------------------------------------------
# Test 8b: Perf #5 — _all_pairs_distance is cached per topology
# ---------------------------------------------------------------------------

def test_perf5_distance_cache_returns_same_object(topo4):
    """Cached _all_pairs_distance returns the SAME dict instance on repeated
    calls with the same topology (Python `is`, not just `==`)."""
    from lccfq_lang.opt.builtin.routing import _all_pairs_distance
    d1 = _all_pairs_distance(topo4)
    d2 = _all_pairs_distance(topo4)
    assert d1 is d2, "expected cached identity-preserved dict"


def test_perf5_distance_cache_distinct_topologies(isa, topo4):
    """Two distinct QPUTopology instances get distinct cache entries."""
    from lccfq_lang.mach.topology import QPUTopology
    from lccfq_lang.opt.builtin.routing import _all_pairs_distance
    # Build a second linear-4 topology independently.
    import networkx as nx
    topo4b = QPUTopology.__new__(QPUTopology)
    topo4b.internal = nx.Graph()
    topo4b.internal.add_edges_from([(0, 1), (1, 2), (2, 3)])
    d1 = _all_pairs_distance(topo4)
    d2 = _all_pairs_distance(topo4b)
    assert d1 is not d2
    # But values should be equal (same shape).
    assert d1 == d2


def test_perf5_distance_cache_hits_increase(topo4):
    """The cache-stats counter records hits after the first compute."""
    from lccfq_lang.opt.builtin.routing import (
        _all_pairs_distance, _DISTANCE_CACHE_STATS,
    )
    before = dict(_DISTANCE_CACHE_STATS)
    _all_pairs_distance(topo4)  # cold or hit depending on prior tests
    _all_pairs_distance(topo4)  # definitely a hit
    _all_pairs_distance(topo4)  # definitely a hit
    after = dict(_DISTANCE_CACHE_STATS)
    # At least 2 hits added (we asked twice after the first guaranteed-cached call).
    assert after["hits"] - before["hits"] >= 2


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
    result, _ = pass_inst.run(list(program), ctx4)

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
    result, _ = pass_inst.run([], ctx4)
    assert result == []


def test_1q_only_no_swaps_mixed_with_measure(isa, topo4, ctx4):
    """1q gates and measure instructions pass through with no SWAPs."""
    program = [
        _make_mapped(isa.h(tg=0)),
        _make_mapped(isa.x(tg=1)),
        _make_mapped(isa.measure(tgs=[0, 1])),
    ]
    pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topo4)
    result, _ = pass_inst.run(program, ctx4)

    assert not any(i.symbol == "swap" for i in result)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Perf #11 new tests (N1–N17)
# ---------------------------------------------------------------------------

# Helper: build a QRegister for testing LayoutSelectionPass.
def _make_qreg(virtual_qubits, topo, isa_inst):
    from lccfq_lang.arch.register import QRegister
    mapping = QPUMapping(virtual_qubits, topo)
    return QRegister(len(virtual_qubits), mapping, isa_inst)


# N1: sabre_fast strategy accepted by mapping
def test_sabre_fast_strategy_accepted_by_mapping(topo4):
    """QPUMapping with routing_strategy='sabre_fast' constructs without error."""
    m = QPUMapping([0, 1, 2, 3], topo4, routing_strategy="sabre_fast")
    assert m.routing_strategy == "sabre_fast"


# N2: _dedup_unique_2q_pairs — basic deduplication and exclusions
def test_dedup_unique_2q_pairs_basic(isa):
    """cx(0,1), cx(0,1), cx(1,2), h(0), measure → [(0,1), (1,2)]."""
    program = [
        isa.cx(ct=0, tg=1),
        isa.cx(ct=0, tg=1),  # duplicate
        isa.cx(ct=1, tg=2),
        isa.h(tg=0),         # 1q — excluded
        isa.measure(tgs=[0]),  # measure — excluded
    ]
    result = _dedup_unique_2q_pairs(program)
    assert result == [(0, 1), (1, 2)]


# N3: _dedup_unique_2q_pairs — order preservation
def test_dedup_unique_2q_pairs_preserves_order(isa):
    """First-occurrence order is preserved; different emission order → different result."""
    program_a = [isa.cx(ct=0, tg=1), isa.cx(ct=1, tg=2)]
    program_b = [isa.cx(ct=1, tg=2), isa.cx(ct=0, tg=1)]
    result_a = _dedup_unique_2q_pairs(program_a)
    result_b = _dedup_unique_2q_pairs(program_b)
    assert result_a == [(0, 1), (1, 2)]
    assert result_b == [(1, 2), (0, 1)]
    # Same set of pairs but different order.
    assert set(result_a) == set(result_b)
    assert result_a != result_b


# N4: _proxy_cost — ranks known fixture correctly
def test_proxy_cost_ranks_known_fixture(topo4):
    """On linear-4, unique_pair (0,3): identity layout (d=3) → cost 2."""
    distances = _all_pairs_distance(topo4)
    # virtual pair (0, 3); layout is virtual->physical
    unique_virtual_pairs = [(0, 3)]
    # Identity layout: virt 0->phys 0, virt 3->phys 3; d(0,3)=3 → cost=2.
    layout_identity = {0: 0, 1: 1, 2: 2, 3: 3}
    cost_identity = _proxy_cost(unique_virtual_pairs, layout_identity, distances)
    assert cost_identity == 2

    # Swap layout: virt 0->phys 0, virt 3->phys 2; d(0,2)=2 → cost=1.
    layout_swap = {0: 0, 1: 1, 2: 3, 3: 2}
    cost_swap = _proxy_cost(unique_virtual_pairs, layout_swap, distances)
    assert cost_swap == 1

    # Optimal layout: virt 0->phys 0, virt 3->phys 1; d(0,1)=1 → cost=0.
    layout_opt = {0: 0, 1: 2, 2: 3, 3: 1}
    cost_opt = _proxy_cost(unique_virtual_pairs, layout_opt, distances)
    assert cost_opt == 0

    # cost_identity > cost_swap > cost_opt
    assert cost_identity > cost_swap >= cost_opt


# N5: _proxy_cost — zero on adjacent pairs
def test_proxy_cost_zero_on_adjacent_pairs(topo4):
    """Pairs already on edges → cost 0."""
    distances = _all_pairs_distance(topo4)
    # On linear-4: (0,1), (1,2), (2,3) are edges (d=1).
    unique_pairs = [(0, 1), (1, 2), (2, 3)]
    layout = {0: 0, 1: 1, 2: 2, 3: 3}
    cost = _proxy_cost(unique_pairs, layout, distances)
    assert cost == 0


# N6: _effective_max_rounds — formula spot checks
def test_effective_max_rounds_formula():
    """Verify adaptive cap formula: max(3, MAX_ROUNDS_DEFAULT // n_qubits)."""
    assert _effective_max_rounds(0) == MAX_ROUNDS_DEFAULT   # n=0 → base
    assert _effective_max_rounds(1) == 50    # 50//1 = 50
    assert _effective_max_rounds(3) == 16    # 50//3 = 16
    assert _effective_max_rounds(5) == 10    # 50//5 = 10
    assert _effective_max_rounds(10) == 5    # 50//10 = 5
    assert _effective_max_rounds(17) == 3    # 50//17 = 2 → floored to 3
    assert _effective_max_rounds(20) == 3    # 50//20 = 2 → floored to 3


# N7: LayoutSelectionPass returns changed=True when layout changes
def test_layout_selection_pass_returns_changed_true_when_layout_changes(isa, topo4):
    """Repeated cx(0,3) fixture: LayoutSelectionPass must find a better layout."""
    from lccfq_lang.opt.builtin.lower_passes import MappedPass

    virtual_program = []
    for _ in range(5):
        virtual_program.append(isa.cx(ct=0, tg=3))
        virtual_program.append(isa.cx(ct=3, tg=0))

    qreg = _make_qreg([0, 1, 2, 3], topo4, isa)
    ctx = PassContext(topology=topo4, isa=isa)

    # First apply MappedPass (virtual->physical).
    mapped_pass = MappedPass(qreg)
    mapped_program, _ = mapped_pass.run(virtual_program, ctx)

    # Then run LayoutSelectionPass.
    layout_pass = LayoutSelectionPass(qreg, isa, topo4, oracle="proxy")
    out_program, changed = layout_pass.run(mapped_program, ctx)

    # The proxy should find a better layout for cx(0,3) on a linear-4.
    assert changed is True
    # The rewritten program should have 2q gates on a closer physical pair.
    out_pairs = {
        (min(i.control_qubits[0], i.target_qubits[0]),
         max(i.control_qubits[0], i.target_qubits[0]))
        for i in out_program
        if _two_qubit_qubits(i) is not None
    }
    # Identity layout puts them at distance 3; any improvement means ≤ distance 2.
    from lccfq_lang.opt.builtin.routing import _all_pairs_distance
    dist = _all_pairs_distance(topo4)
    max_dist = max(dist.get((a, b), 0) for (a, b) in out_pairs) if out_pairs else 0
    assert max_dist <= 2, (
        f"Expected improved layout with max distance ≤ 2, got {max_dist}"
    )


# N8: LayoutSelectionPass returns changed=False when layout already optimal
def test_layout_selection_pass_returns_changed_false_when_layout_optimal(isa, topo4):
    """cx(0,1)+cx(1,2) on linear-4 with identity mapping — already optimal."""
    from lccfq_lang.opt.builtin.lower_passes import MappedPass

    virtual_program = [isa.cx(ct=0, tg=1), isa.cx(ct=1, tg=2)]
    qreg = _make_qreg([0, 1, 2, 3], topo4, isa)
    ctx = PassContext(topology=topo4, isa=isa)

    mapped_pass = MappedPass(qreg)
    mapped_program, _ = mapped_pass.run(virtual_program, ctx)

    layout_pass = LayoutSelectionPass(qreg, isa, topo4, oracle="proxy")
    out_program, changed = layout_pass.run(mapped_program, ctx)

    # Adjacent pairs on identity layout are already cost=0; no change expected.
    assert changed is False


# N9: LayoutSelectionPass is first in lower_swap when sabre_fast
def test_layout_selection_pass_is_first_in_lower_swap_when_sabre_fast(topo4, isa):
    """build_lowering_groups with sabre_fast: lower_swap[0].name == 'layout_selection'."""
    from pathlib import Path
    from lccfq_lang.backend import QPU
    from lccfq_lang.opt.builtin.lower_passes import build_lowering_groups

    qpu = QPU(
        filename=str(Path(__file__).parent.parent / "data" / "testing.toml"),
        last_pass="swapped",
    )
    qreg = qpu.qregister(4)
    groups = build_lowering_groups(qreg, qpu, routing_strategy="sabre_fast")
    lower_swap = next(g for g in groups if g.name == "lower_swap")
    assert lower_swap.passes[0].name == "layout_selection"
    assert lower_swap.passes[1].name == "swapped"


# N10: LayoutSelectionPass is first in lower_swap when sabre_lite
def test_layout_selection_pass_is_first_in_lower_swap_when_sabre_lite(topo4, isa):
    """build_lowering_groups with sabre_lite: lower_swap[0].name == 'layout_selection'."""
    from pathlib import Path
    from lccfq_lang.backend import QPU
    from lccfq_lang.opt.builtin.lower_passes import build_lowering_groups

    qpu = QPU(
        filename=str(Path(__file__).parent.parent / "data" / "testing.toml"),
        last_pass="swapped",
    )
    qreg = qpu.qregister(4)
    groups = build_lowering_groups(qreg, qpu, routing_strategy="sabre_lite")
    lower_swap = next(g for g in groups if g.name == "lower_swap")
    assert lower_swap.passes[0].name == "layout_selection"
    assert lower_swap.passes[1].name == "swapped"


# N11: LayoutSelectionPass absent when strategy is identity
def test_layout_selection_pass_absent_when_identity(topo4, isa):
    """With routing_strategy='identity', lower_swap contains only SwappedPass."""
    from pathlib import Path
    from lccfq_lang.backend import QPU
    from lccfq_lang.opt.builtin.lower_passes import build_lowering_groups

    qpu = QPU(
        filename=str(Path(__file__).parent.parent / "data" / "testing.toml"),
        last_pass="swapped",
    )
    qreg = qpu.qregister(4)
    groups = build_lowering_groups(qreg, qpu, routing_strategy="identity")
    lower_swap = next(g for g in groups if g.name == "lower_swap")
    assert len(lower_swap.passes) == 1
    assert lower_swap.passes[0].name == "swapped"


# N12: LayoutSelectionPass records scratchpad keys
def test_layout_selection_pass_records_scratchpad_keys(isa, topo4):
    """After running LayoutSelectionPass with a non-identity result,
    ctx.scratchpad must contain layout_selection.new_layout and
    layout_selection.permutation."""
    from lccfq_lang.opt.builtin.lower_passes import MappedPass

    virtual_program = []
    for _ in range(5):
        virtual_program.append(isa.cx(ct=0, tg=3))
        virtual_program.append(isa.cx(ct=3, tg=0))

    qreg = _make_qreg([0, 1, 2, 3], topo4, isa)
    ctx = PassContext(topology=topo4, isa=isa)

    mapped_pass = MappedPass(qreg)
    mapped_program, _ = mapped_pass.run(virtual_program, ctx)

    layout_pass = LayoutSelectionPass(qreg, isa, topo4, oracle="proxy")
    _, changed = layout_pass.run(mapped_program, ctx)

    if changed:
        assert "layout_selection.new_layout" in ctx.scratchpad
        assert "layout_selection.permutation" in ctx.scratchpad
        nl = ctx.scratchpad["layout_selection.new_layout"]
        perm = ctx.scratchpad["layout_selection.permutation"]
        # new_layout is virtual->physical; perm is phys->phys
        assert set(nl.keys()) == {0, 1, 2, 3}
        assert set(perm.keys()).issubset(set(topo4.qubits()))
    else:
        # If layout didn't change (already optimal), scratchpad is not populated.
        pass


# N13: end-to-end Circuit opt_level=2 uses sabre_fast
def test_end_to_end_circuit_opt_level_2_uses_sabre_fast():
    """Circuit(opt_level=2, report=True): opt_report['routing_strategy'] == 'sabre_fast'
    and lower_swap group has passes[0].name == 'layout_selection'."""
    from pathlib import Path
    from lccfq_lang.backend import QPU
    from lccfq_lang.arch.context import Circuit
    from lccfq_lang.arch.register import QRegister, CRegister

    qpu = QPU(
        filename=str(Path(__file__).parent.parent / "data" / "testing.toml"),
        last_pass="swapped",
    )
    qreg = qpu.qregister(4)
    creg = CRegister(4)

    with Circuit(qreg, creg, qpu=qpu, opt_level=2, report=True) as c:
        c >> qpu.isa.cx(ct=0, tg=3)
        c >> qpu.isa.cx(ct=3, tg=0)

    report = c.opt_report
    assert report is not None
    assert report["routing_strategy"] == "sabre_fast"

    # Find the lower_swap group.
    lower_swap_group = next(
        (g for g in report["groups"] if g["name"] == "lower_swap"),
        None,
    )
    assert lower_swap_group is not None
    assert lower_swap_group["passes"][0]["name"] == "layout_selection"


# N14: explicit sabre_lite preserved at opt_level=1
def test_end_to_end_circuit_explicit_sabre_lite_preserved():
    """QPUMapping with sabre_lite + opt_level=1: stays sabre_lite."""
    from pathlib import Path
    from lccfq_lang.backend import QPU
    from lccfq_lang.arch.context import Circuit
    from lccfq_lang.arch.register import QRegister, CRegister
    from lccfq_lang.arch.mapping import QPUMapping
    from lccfq_lang.mach.topology import QPUTopology
    from lccfq_lang.sys.base import QPUConfig
    import toml

    qpu = QPU(
        filename=str(Path(__file__).parent.parent / "data" / "testing.toml"),
        last_pass="swapped",
    )
    # Manually create a qreg with sabre_lite strategy.
    from lccfq_lang.arch.register import QRegister
    mapping = QPUMapping([0, 1, 2, 3], qpu.mapping.topology, routing_strategy="sabre_lite")
    from lccfq_lang.arch.register import QRegister
    qreg = QRegister(4, mapping, qpu.isa)
    creg = CRegister(4)

    with Circuit(qreg, creg, qpu=qpu, opt_level=1, report=True) as c:
        c >> qpu.isa.cx(ct=0, tg=3)
        c >> qpu.isa.cx(ct=3, tg=0)

    report = c.opt_report
    assert report is not None
    assert report["routing_strategy"] == "sabre_lite"


# N15: layout improvement via LayoutSelectionPass ≤ legacy + slack
def test_layout_improvement_via_pass_matches_legacy_sabre_lite(isa, topo4):
    """Repeated cx(0,3) fixture: sabre_fast SWAP count ≤ sabre_lite SWAP count + 2."""
    from lccfq_lang.opt.builtin.lower_passes import MappedPass, build_lowering_groups
    from lccfq_lang.opt.manager import PassManager
    from pathlib import Path
    from lccfq_lang.backend import QPU

    qpu = QPU(
        filename=str(Path(__file__).parent.parent / "data" / "testing.toml"),
        last_pass="swapped",
    )

    virtual_program = []
    for _ in range(5):
        virtual_program.append(isa.cx(ct=0, tg=3))
        virtual_program.append(isa.cx(ct=3, tg=0))

    def _count_swaps_for_strategy(strategy):
        qreg = qpu.qregister(4)
        ctx = PassContext(
            qpu_config=qpu.config,
            isa=qpu.isa,
            mapping=qreg.mapping,
            topology=topo4,
        )
        groups = build_lowering_groups(qreg, qpu, routing_strategy=strategy)
        # Only run up to lower_swap.
        from lccfq_lang.opt.builtin.lower_passes import slice_groups_for
        groups = slice_groups_for("swapped", groups)
        program, _, _ = PassManager(groups).run(list(virtual_program), ctx)
        return sum(1 for i in program if i.symbol == "swap")

    fast_swaps = _count_swaps_for_strategy("sabre_fast")
    lite_swaps = _count_swaps_for_strategy("sabre_lite")

    # Regression guard: proxy should not be dramatically worse.
    assert fast_swaps <= lite_swaps + 2, (
        f"sabre_fast SWAP count ({fast_swaps}) exceeded sabre_lite ({lite_swaps}) + 2"
    )


# N16: opt_report lower_swap group has two passes under sabre_fast
def test_opt_report_lower_swap_group_has_two_passes_under_sabre_fast():
    """Direct check: len(lower_swap['passes']) == 2 under sabre_fast."""
    from pathlib import Path
    from lccfq_lang.backend import QPU
    from lccfq_lang.arch.context import Circuit
    from lccfq_lang.arch.register import CRegister

    qpu = QPU(
        filename=str(Path(__file__).parent.parent / "data" / "testing.toml"),
        last_pass="swapped",
    )
    qreg = qpu.qregister(4)
    creg = CRegister(4)

    with Circuit(qreg, creg, qpu=qpu, opt_level=2, report=True) as c:
        c >> qpu.isa.cx(ct=0, tg=3)
        c >> qpu.isa.cx(ct=0, tg=2)

    report = c.opt_report
    lower_swap_group = next(
        (g for g in report["groups"] if g["name"] == "lower_swap"),
        None,
    )
    assert lower_swap_group is not None
    assert len(lower_swap_group["passes"]) == 2, (
        f"Expected 2 passes in lower_swap under sabre_fast, "
        f"got {len(lower_swap_group['passes'])}"
    )
