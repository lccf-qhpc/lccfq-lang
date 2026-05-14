"""
Filename: routing.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.0
Description:
    SABRE-lite circuit-aware routing pass and trivial-then-improve
    layout selection for the lccfq-lang lowering pipeline.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import math
from itertools import combinations
from typing import List, Optional
import networkx as nx

from lccfq_lang.arch.instruction import Instruction, InstructionType
from lccfq_lang.arch.isa import ISA
from lccfq_lang.arch.register import QRegister
from lccfq_lang.mach.topology import QPUTopology
from lccfq_lang.opt.pass_base import Pass, PassContext


# ---------------------------------------------------------------------------
# Tunable constants (§4.2)
# ---------------------------------------------------------------------------

ALPHA: float = 0.5
DECAY_WEIGHT: float = 0.001
LOOKAHEAD_K: int = 20


def _stall_cap(topology: QPUTopology) -> int:
    """Hard iteration cap for stall detection: 8 * number of qubits."""
    return 8 * len(topology.qubits())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_qubit_qubits(instr: Instruction) -> Optional[tuple]:
    """Return (q0, q1) iff this is a routable 2q gate, else None.

    Measure and reset are excluded. A 2q gate has exactly 2 total qubits
    across control and target lists.
    """
    if instr.symbol in ("measure", "reset"):
        return None
    targets = instr.target_qubits or []
    controls = instr.control_qubits or []
    qs = list(controls) + list(targets)
    if len(qs) != 2:
        return None
    return (qs[0], qs[1])


def _all_pairs_distance(topology: QPUTopology) -> dict:
    """Pre-compute BFS all-pairs distances over the topology graph.

    Returns a dict keyed by (src, dst) -> int hop count.
    Same-qubit distance is 0. Unreachable pairs are math.inf.
    """
    g: nx.Graph = topology.internal
    dist: dict = {}
    for src in g.nodes:
        lengths = nx.single_source_shortest_path_length(g, src)
        for dst, d in lengths.items():
            dist[(src, dst)] = d
    return dist


def _bfs_distance(topology: QPUTopology, src: int, dst: int) -> int:
    """Return BFS hop distance between src and dst on the topology.

    Returns math.inf if no path exists.
    """
    if src == dst:
        return 0
    try:
        return nx.shortest_path_length(topology.internal, src, dst)
    except nx.NetworkXNoPath:
        return math.inf


def _rewrite(instr: Instruction, layout: dict) -> Instruction:
    """Return a copy of instr with qubit indices mapped through layout.

    All non-qubit attributes are preserved. The returned instruction
    has is_mapped=True.

    :param instr: source instruction (not mutated)
    :param layout: phys->phys permutation mapping original physical id to current
    :return: new Instruction with rewritten qubits
    """
    new_targets = (
        [layout[q] for q in instr.target_qubits] if instr.target_qubits else []
    )
    new_controls = (
        [layout[q] for q in instr.control_qubits] if instr.control_qubits else []
    )
    out = Instruction(
        symbol=instr.symbol,
        modifies_state=instr.modifies_state,
        is_controlled=instr.is_controlled,
        target_qubits=new_targets,
        control_qubits=new_controls,
        params=instr.params,
        shots=instr.shots,
    )
    out.instruction_type = InstructionType.DELAYED
    out.pre = instr.pre.copy()
    out.post = instr.post.copy()
    out.is_mapped = True
    return out


def _score_swap(
    swap_pair: tuple,
    front: list,
    lookahead: list,
    current_layout: dict,
    distances: dict,
    decay: dict,
    alpha: float = ALPHA,
    decay_weight: float = DECAY_WEIGHT,
) -> float:
    """Compute the SABRE-lite score for a candidate SWAP.

    Lower score is better. Negative scores indicate the swap brings
    gates closer. Decay penalty discourages re-using recently-swapped
    qubits to prevent oscillation.

    :param swap_pair: (a, b) physical qubit pair (a < b)
    :param front: front layer of non-adjacent 2q gates
    :param lookahead: next K 2q gates beyond the front layer
    :param current_layout: phys->phys permutation
    :param distances: pre-computed all-pairs BFS distances
    :param decay: per-qubit accumulated decay penalty
    :param alpha: weight of lookahead vs front layer
    :param decay_weight: per-SWAP per-qubit decay increment
    :return: float score (lower is better)
    """
    a, b = swap_pair

    def gate_dist(instr: Instruction, layout: dict) -> float:
        pair = _two_qubit_qubits(instr)
        if pair is None:
            return 0
        p0 = layout[pair[0]]
        p1 = layout[pair[1]]
        return distances.get((p0, p1), math.inf)

    # Build hypothetical layout after applying this SWAP.
    # current_layout is phys->phys: swap the values at keys whose
    # current values are a and b respectively.
    hypo = dict(current_layout)
    inv = {v: k for k, v in hypo.items()}
    ka, kb = inv[a], inv[b]
    hypo[ka], hypo[kb] = b, a

    delta_front = sum(
        gate_dist(g, hypo) - gate_dist(g, current_layout) for g in front
    )
    delta_look = sum(
        gate_dist(g, hypo) - gate_dist(g, current_layout) for g in lookahead
    )
    penalty = decay_weight * (decay.get(a, 0.0) + decay.get(b, 0.0))
    return delta_front + alpha * delta_look + penalty


# ---------------------------------------------------------------------------
# LookaheadSwapInsertion Pass
# ---------------------------------------------------------------------------

class LookaheadSwapInsertion(Pass):
    """SABRE-lite routing pass.

    Replaces SwappedPass when routing_strategy == "sabre_lite". Operates on
    a mapped (physical-qubit) program; emits a new program with SWAPs
    inserted to satisfy adjacency constraints.

    When used as a cost oracle for LayoutSelection, qreg may be None.
    The routing logic does not require the register — only isa and topology.
    """

    name = "swapped"
    applies_to = "arch"

    def __init__(
        self,
        qreg: Optional[QRegister],
        isa: ISA,
        topology: QPUTopology,
    ) -> None:
        """Create the routing pass.

        :param qreg: quantum register (may be None for dry-run cost oracle use)
        :param isa: instruction set architecture, used to construct SWAP gates
        :param topology: QPU connectivity graph
        """
        self._qreg = qreg
        self._isa = isa
        self._topology = topology

    def run(self, program: List[Instruction], ctx: PassContext) -> List[Instruction]:
        """Route the program by inserting SWAPs as needed.

        :param program: list of already-mapped (physical qubit) Instructions
        :param ctx: pass context (topology read from self._topology)
        :return: new instruction list with SWAP gates inserted
        """
        if not program:
            return list(program)

        topology = self._topology
        distances = _all_pairs_distance(topology)
        emitted: list = []
        queue = list(program)

        # Identity permutation: current_layout[p] = p for all physical qubits.
        current_layout: dict = {p: p for p in topology.qubits()}
        decay: dict = {p: 0.0 for p in topology.qubits()}

        stall_iters = 0
        last_head_dist = None
        cap = _stall_cap(topology)

        while queue:
            # Step 1: Drain leading 1q / measure / reset / adjacent 2q gates.
            progressed = True
            while progressed and queue:
                progressed = False
                head = queue[0]
                pair = _two_qubit_qubits(head)
                if pair is None:
                    # 1q gate, measure, or reset: emit with qubit rewrite.
                    emitted.append(_rewrite(head, current_layout))
                    queue.pop(0)
                    progressed = True
                    continue
                # 2q gate: check adjacency under current layout.
                p0 = current_layout[pair[0]]
                p1 = current_layout[pair[1]]
                if topology.internal.has_edge(p0, p1):
                    emitted.append(_rewrite(head, current_layout))
                    queue.pop(0)
                    progressed = True

            if not queue:
                break

            # Step 2: Head is a non-adjacent 2q gate.
            head = queue[0]
            head_pair = _two_qubit_qubits(head)
            front = [head]

            # Build lookahead: next LOOKAHEAD_K 2q gates in the queue.
            lookahead: list = []
            for instr in queue[1:]:
                if _two_qubit_qubits(instr) is not None:
                    lookahead.append(instr)
                    if len(lookahead) >= LOOKAHEAD_K:
                        break

            # Step 3: Build candidate SWAP set — topology edges incident to
            # the physical qubits currently used by the front layer.
            front_phys = {
                current_layout[head_pair[0]],
                current_layout[head_pair[1]],
            }
            cand: list = sorted(
                (
                    (min(u, v), max(u, v))
                    for (u, v) in topology.internal.edges
                    if u in front_phys or v in front_phys
                ),
                key=lambda p: (p[0], p[1]),
            )
            # Safety fallback: if no incident edges (e.g., qubit isolated in
            # subgraph), expand to all topology edges.
            if not cand:
                cand = sorted(
                    ((min(u, v), max(u, v)) for (u, v) in topology.internal.edges),
                    key=lambda p: (p[0], p[1]),
                )

            # If still no candidates, the topology has no edges at all — the
            # gate is permanently unmappable.
            if not cand:
                raise RuntimeError(
                    f"LookaheadSwapInsertion: program contains an "
                    f"unmappable 2q gate on qubits {head_pair}"
                )

            # Steps 4+5: Score candidates and pick best (ascending score,
            # tie-break by lowest (min, max) pair).
            best_swap = None
            best_score = math.inf
            for sp in cand:
                s = _score_swap(
                    sp, front, lookahead, current_layout, distances, decay,
                )
                if s < best_score or (
                    s == best_score
                    and (best_swap is None or sp < best_swap)
                ):
                    best_score = s
                    best_swap = sp

            assert best_swap is not None, "Unreachable: cand is non-empty but no best_swap selected"

            # Step 6: Emit the SWAP and update state.
            a, b = best_swap
            swap_instr = self._isa.swap(tg_a=a, tg_b=b)
            swap_instr.is_mapped = True
            emitted.append(swap_instr)

            # Update permutation: swap values at the keys that currently
            # hold a and b.
            inv = {v: k for k, v in current_layout.items()}
            ka, kb = inv[a], inv[b]
            current_layout[ka], current_layout[kb] = b, a

            decay[a] = decay.get(a, 0.0) + DECAY_WEIGHT
            decay[b] = decay.get(b, 0.0) + DECAY_WEIGHT

            # Stall detection: track distance of head 2q gate.
            new_dist = distances.get(
                (current_layout[head_pair[0]], current_layout[head_pair[1]]),
                math.inf,
            )
            if last_head_dist is not None and new_dist >= last_head_dist:
                stall_iters += 1
            else:
                stall_iters = 0
            last_head_dist = new_dist

            if stall_iters > cap:
                raise RuntimeError(
                    f"LookaheadSwapInsertion: program contains an "
                    f"unmappable 2q gate on qubits {head_pair}"
                )

        return emitted


# ---------------------------------------------------------------------------
# LayoutSelection
# ---------------------------------------------------------------------------

class LayoutSelection:
    """Trivial-then-improve layout chooser.

    Runs before the lowering pipeline when routing_strategy == "sabre_lite".
    Returns a virtual->physical mapping that minimises SWAP count as
    predicted by LookaheadSwapInsertion.
    """

    @staticmethod
    def compute_layout(
        program: List[Instruction],
        topology: QPUTopology,
        isa: ISA,
        initial_layout: dict,
        max_rounds: int = 50,
        patience: int = 10,
    ) -> dict:
        """Find an improved virtual->physical layout by local swap search.

        Uses a trivial-then-improve strategy: start from initial_layout and
        repeatedly try swapping pairs of virtual qubits. Accept strictly
        better layouts only (lower SWAP count). Stop when patience
        non-improving rounds have passed or max_rounds is reached.

        :param program: virtual-qubit instruction list (pre-MappedPass)
        :param topology: QPU connectivity
        :param isa: ISA for SWAP construction in dry runs
        :param initial_layout: virtual->physical mapping to start from
        :param max_rounds: maximum improvement rounds
        :param patience: early-stop after this many non-improving rounds
        :return: best virtual->physical mapping found
        """
        if not program:
            return dict(initial_layout)

        best = dict(initial_layout)
        best_swaps = LayoutSelection._count_swaps(program, best, topology, isa)
        no_improve = 0

        virtual_keys = sorted(initial_layout.keys())
        pairs: list = sorted(
            combinations(virtual_keys, 2),
            key=lambda p: (p[0], p[1]),
        )

        for _ in range(max_rounds):
            improved = False
            for (va, vb) in pairs:
                cand = dict(best)
                cand[va], cand[vb] = cand[vb], cand[va]
                c_swaps = LayoutSelection._count_swaps(program, cand, topology, isa)
                if c_swaps < best_swaps:
                    best, best_swaps = cand, c_swaps
                    improved = True
                    break  # restart pair scan from the top
            if not improved:
                no_improve += 1
                if no_improve >= patience:
                    break
            else:
                no_improve = 0

        return best

    @staticmethod
    def _count_swaps(
        program: List[Instruction],
        layout: dict,
        topology: QPUTopology,
        isa: ISA,
    ) -> int:
        """Count SWAPs inserted by LookaheadSwapInsertion under a candidate layout.

        Simulates MappedPass (applying layout to virtual qubits) then runs
        LookaheadSwapInsertion in cost-oracle mode (qreg=None).

        :param program: virtual-qubit instruction list
        :param layout: virtual->physical candidate mapping
        :param topology: QPU connectivity
        :param isa: ISA for SWAP construction
        :return: number of SWAP instructions in the routed output
        """
        # Simulate MappedPass: rewrite virtual qubit ids to physical ids.
        mapped: list = []
        for instr in program:
            new_targets = (
                [layout[q] for q in instr.target_qubits]
                if instr.target_qubits
                else []
            )
            new_controls = (
                [layout[q] for q in instr.control_qubits]
                if instr.control_qubits
                else []
            )
            m = Instruction(
                symbol=instr.symbol,
                modifies_state=instr.modifies_state,
                is_controlled=instr.is_controlled,
                target_qubits=new_targets,
                control_qubits=new_controls,
                params=instr.params,
                shots=instr.shots,
            )
            m.instruction_type = InstructionType.DELAYED
            m.pre = instr.pre.copy()
            m.post = instr.post.copy()
            m.is_mapped = True
            mapped.append(m)

        # Run routing in cost-oracle mode.
        pass_inst = LookaheadSwapInsertion(qreg=None, isa=isa, topology=topology)
        routed = pass_inst.run(mapped, PassContext(topology=topology, isa=isa))
        return sum(1 for ins in routed if ins.symbol == "swap")
