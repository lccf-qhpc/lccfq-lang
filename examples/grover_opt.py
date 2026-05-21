"""
Filename: grover_opt.py
Author: Santiago Nunez-Corrales
Date: 2026-05-15
Version: 1.1
Description:
    Parametric Grover search demo. Showcases the effect of each optimization
    level (0, 1, 2, 3) on a non-trivial circuit built from the PHASE_ORACLE
    and DIFFUSION blocks for an arbitrary search-register width.

    Multi-controlled gates are decomposed by the multicontrol block. The
    mode is selectable via --mc-mode:
        vchain   ancilla-free, O(n^2) two-qubit count
        barenco  needs a workspace ancilla, O(n) two-qubit count

    Optimal Grover iterations for n qubits is round(pi/4 * sqrt(2^n)). The
    demo defaults to 1 iteration so compile time stays modest; raise it
    with --iterations once you have a feel for the optimization deltas.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import argparse
import math
import sys
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


def validate_configuration(
    qpu: QPU,
    n_qubits: int,
    marked_state: int,
    mc_mode: str,
    workspace: Optional[int],
) -> None:
    """Validate user inputs against the QPU config and the MC mode contract.

    Raises SystemExit with a descriptive message on the first violation,
    so the user sees a single clear error rather than a stack trace from
    deep inside the compiler.
    """
    qpu_count = qpu.config.qubit_count

    # Search-register width sanity.
    if n_qubits < 2:
        raise SystemExit(
            f"--n-qubits must be at least 2 (Grover needs a search register); "
            f"got {n_qubits}."
        )
    if n_qubits > qpu_count:
        raise SystemExit(
            f"--n-qubits={n_qubits} exceeds the QPU's qubit_count={qpu_count}. "
            f"Either lower --n-qubits or bump qubit_count in the QPU config."
        )

    # Marked state must be addressable by the search register.
    if marked_state < 0 or marked_state >= (1 << n_qubits):
        raise SystemExit(
            f"--marked-state={marked_state} is out of range for n_qubits={n_qubits} "
            f"(valid range is 0..{(1 << n_qubits) - 1})."
        )

    # MC-mode-driven workspace requirements.
    # PHASE_ORACLE and DIFFUSION emit mcz with (n_qubits - 1) controls. The
    # underlying multicontrol block only needs an ancilla when n_controls >= 2,
    # i.e. when n_qubits >= 3.
    n_controls = n_qubits - 1
    needs_ancilla = (mc_mode == "barenco" and n_controls >= 2)

    if mc_mode == "vchain":
        if workspace is not None:
            print(
                f"warning: --workspace={workspace} ignored in vchain mode "
                f"(vchain decomposes without an ancilla).",
                file=sys.stderr,
            )
        # vchain itself does not consume an ancilla, but the spec allocates
        # internal scratch within the search register, so no extra check.
        return

    # mc_mode == "barenco"
    if not needs_ancilla:
        # n_qubits == 2 (n_controls == 1) reduces to a plain cz: no ancilla needed.
        if workspace is not None:
            print(
                f"warning: --workspace={workspace} ignored at n_qubits={n_qubits} "
                f"in barenco mode (no MC decomposition is triggered).",
                file=sys.stderr,
            )
        return

    if workspace is None:
        raise SystemExit(
            f"barenco mode with n_qubits={n_qubits} requires --workspace. "
            f"Provide a qubit index in [{n_qubits}..{qpu_count - 1}] "
            f"(and ensure the QPU config has qubit_count >= {n_qubits + 1})."
        )

    if workspace < 0 or workspace >= qpu_count:
        raise SystemExit(
            f"--workspace={workspace} is outside the QPU's qubit range "
            f"[0..{qpu_count - 1}]. Bump qubit_count in the QPU config "
            f"or pick another index."
        )

    if workspace < n_qubits:
        raise SystemExit(
            f"--workspace={workspace} collides with the search register "
            f"[0..{n_qubits - 1}]. Pick an index in [{n_qubits}..{qpu_count - 1}]."
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parametric Grover search demo showing opt-level effects.",
    )
    p.add_argument(
        "--n-qubits", "-n", type=int, default=6,
        help=(
            "Search-register width in qubits (default 6). Must satisfy "
            "2 <= n <= qpu.qubit_count. For barenco mode an additional "
            "workspace qubit is required, so qubit_count must be at least n+1."
        ),
    )
    p.add_argument(
        "--config", type=str, default="config/grover.toml",
        help="QPU configuration file (default config/grover.toml).",
    )
    p.add_argument(
        "--iterations", "-i", type=int, default=1,
        help=(
            "Grover iterations (default 1). Optimal for n qubits is "
            "round(pi/4 * sqrt(2^n)); compile time grows linearly with this."
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
            "Ancilla qubit index for barenco mode. Must lie outside the "
            "search register and inside the QPU's qubit range. Ignored in "
            "vchain mode."
        ),
    )
    args = p.parse_args()

    qpu = QPU(filename=args.config)

    validate_configuration(
        qpu, args.n_qubits, args.marked_state, args.mc_mode, args.workspace,
    )

    n_qubits = args.n_qubits
    optimal_iters = round(math.pi / 4 * math.sqrt(2 ** n_qubits))
    width = max(n_qubits, 1)

    print(f"=== {n_qubits}-qubit Grover demo ===")
    print(f"  config          : {args.config} (qubit_count={qpu.config.qubit_count})")
    print(f"  marked state    : {args.marked_state} (0b{args.marked_state:0{width}b})")
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
    table_width = 100
    print("=" * table_width)
    header = (
        f"{'level':>5}  {'depth_in':>9}  {'depth_out':>10}  "
        f"{'2q_in':>6}  {'2q_out':>7}  "
        f"{'score_in':>10}  {'score_out':>10}  "
        f"{'pass_s':>8}  {'wall_s':>8}"
    )
    print(header)
    print("-" * table_width)

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

    print("=" * table_width)


if __name__ == "__main__":
    main()
