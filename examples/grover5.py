"""
Filename: grover5.py
Author: Santiago Nunez-Corrales
Date: 2026-05-15
Version: 1.0
Description:
    5-qubit Grover search demo. Showcases the effect of each optimization
    level (0, 1, 2, 3) on a non-trivial circuit built from the PHASE_ORACLE
    and DIFFUSION blocks.

    Multi-controlled gates are decomposed by the new multicontrol block,
    defaulting to V-chain mode (no ancilla required). The mode can be
    overridden via --mc-mode and the workspace ancilla via --workspace.

    Optimal Grover iterations for n=5 is round(pi/4 * sqrt(2^5)) = 4, but
    the demo defaults to 1 iteration so compile time stays modest. The
    optimization deltas are illustrative regardless of iteration count;
    crank --iterations up if you want to see the full search behavior.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import argparse
import math
import time
from typing import Optional

from lccfq_lang import QPU, CRegister, Circuit
from lccfq_lang.lang.blocks import BlockFactory, BlockType


def build_grover(
    c: Circuit,
    qpu: QPU,
    factory: BlockFactory,
    n_qubits: int,
    marked_state: int,
    n_iterations: int,
    mc_mode: str,
    workspace: Optional[int],
) -> None:
    """Push a Grover circuit into the open Circuit context."""
    target = list(range(n_qubits))

    # Initial uniform superposition over the search register.
    for q in target:
        c >> qpu.isa.h(tg=q)

    # Grover iterations: phase-flip the marked state, then reflect about uniform.
    for _ in range(n_iterations):
        for instr in factory.block(
            BlockType.PHASE_ORACLE,
            target,
            predicate=[marked_state],
            mc_mode=mc_mode,
            workspace=workspace,
        ):
            c >> instr

        for instr in factory.block(
            BlockType.DIFFUSION,
            target,
            mc_mode=mc_mode,
            workspace=workspace,
        ):
            c >> instr

    c >> qpu.isa.measure(tgs=target)


def run_at_level(
    qpu: QPU,
    n_qubits: int,
    marked_state: int,
    n_iterations: int,
    opt_level: int,
    mc_mode: str,
    workspace: Optional[int],
) -> tuple[dict, float]:
    qreg = qpu.qregister(n_qubits)
    creg = CRegister(n_qubits)
    factory = BlockFactory(qreg, creg)

    t0 = time.perf_counter()

    with Circuit(qreg, creg, qpu, opt_level=opt_level, report=True) as c:
        build_grover(
            c, qpu, factory, n_qubits, marked_state, n_iterations,
            mc_mode, workspace,
        )
    elapsed = time.perf_counter() - t0
    return c.opt_report, elapsed


def summarize(report: dict, level: int, wall_seconds: float) -> None:
    before = report["totals"]["cost_before"]
    after = report["totals"]["cost_after"]
    print(f"--- opt_level={level} ---")
    print(f"  routing_strategy : {report['routing_strategy']}")
    print(f"  last_pass        : {report['last_pass']}")
    print(
        f"  before : depth={before['depth']:6d}  1q={before['count_1q']:6d}  "
        f"2q={before['count_2q']:6d}  native2q={before['count_native_2q']:6d}  "
        f"score={before['scalarized']:8.2f}"
    )
    print(
        f"  after  : depth={after['depth']:6d}  1q={after['count_1q']:6d}  "
        f"2q={after['count_2q']:6d}  native2q={after['count_native_2q']:6d}  "
        f"score={after['scalarized']:8.2f}"
    )
    print(f"  scalarized delta : {report['totals']['scalarized_delta']:+10.2f}")
    print(f"  pass-time secs   : {report['totals']['total_seconds']:.4f}")
    print(f"  wall-clock secs  : {wall_seconds:.2f}")
    groups = ", ".join(
        f"{g['name']}({g['iterations']}it)" for g in report["groups"]
    )
    print(f"  groups run       : {groups}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="10-qubit Grover search demo showing opt-level effects.",
    )
    p.add_argument(
        "--iterations", "-i", type=int, default=1,
        help=(
            "Grover iterations (default 1). Optimal for n=5 is 4; compile "
            "time grows linearly with this value."
        ),
    )
    p.add_argument(
        "--marked-state", type=int, default=0b10101,
        help="Marked basis state as an integer (default 21 = 0b10101).",
    )
    p.add_argument(
        "--mc-mode", choices=["vchain", "barenco"], default="vchain",
        help=(
            "Multi-control decomposition mode. vchain is ancilla-free but "
            "O(n^2). barenco needs a workspace qubit but is O(n)."
        ),
    )
    p.add_argument(
        "--workspace", type=int, default=None,
        help=(
            "Ancilla qubit index for barenco mode. Must not be in [0..n-1]. "
            "Requires bumping the QPU config qubit_count to include it."
        ),
    )
    args = p.parse_args()

    n_qubits = 5
    qpu = QPU(filename="config/grover5.toml")
    optimal_iters = round(math.pi / 4 * math.sqrt(2 ** n_qubits))

    print("=== 5-qubit Grover demo ===")
    print(f"  marked state    : {args.marked_state} (0b{args.marked_state:05b})")
    print(f"  iterations      : {args.iterations}  (optimal would be {optimal_iters})")
    print(f"  mc_mode         : {args.mc_mode}")
    print(f"  workspace       : {args.workspace}")
    print()

    reports: dict[int, tuple[dict, float]] = {}

    for level in (0, 1, 2, 3):
        print(f"Compiling at opt_level={level}...", end=" ", flush=True)
        report, wall = run_at_level(
            qpu, n_qubits, args.marked_state, args.iterations,
            level, args.mc_mode, args.workspace,
        )
        print(f"done ({wall:.2f}s wall)")
        reports[level] = (report, wall)

    print()

    for level in (0, 1, 2, 3):
        report, wall = reports[level]
        summarize(report, level, wall)
        print()

    # Comparison table
    width = 100
    print("=" * width)
    header = (
        f"{'level':>5}  {'depth_in':>9}  {'depth_out':>10}  "
        f"{'2q_in':>6}  {'2q_out':>7}  "
        f"{'score_in':>10}  {'score_out':>10}  "
        f"{'pass_s':>8}  {'wall_s':>8}"
    )
    print(header)
    print("-" * width)

    for level in (0, 1, 2, 3):
        report, wall = reports[level]
        b = report["totals"]["cost_before"]
        a = report["totals"]["cost_after"]
        print(
            f"{level:>5}  {b['depth']:>9}  {a['depth']:>10}  "
            f"{b['count_2q']:>6}  {a['count_2q']:>7}  "
            f"{b['scalarized']:>10.2f}  {a['scalarized']:>10.2f}  "
            f"{report['totals']['total_seconds']:>8.3f}  {wall:>8.2f}"
        )

    print("=" * width)


if __name__ == "__main__":
    main()
