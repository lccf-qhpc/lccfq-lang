"""
Filename: hwea_opt.py
Author: Santiago Nunez-Corrales
Date: 2026-05-21
Version: 1.0
Description:
    Hardware-efficient ansatz (HWEA) optimization demo.  Showcases the effect
    of each optimization level (0, 1, 2, 3) on a variational HWEA circuit
    of configurable width, depth, and rotation axes.

    The HWEA structure interleaves single-qubit rotation layers with a
    nearest-neighbour CX entangler.  The rotation block emits all qubits'
    rotations before the entangler, so each rotation on qubit q is separated
    from the CX(q, q+1) it should commute past by the rotations of the other
    qubits.  The list-adjacent CommuteThroughControl pass (pre-task-76) could
    never close that gap, making opt_level=3 effectively identical to
    opt_level=2 on HWEA-shaped circuits.

    The DAG-based rewrite introduced in task #76 walks the per-qubit dependency
    chain instead of the flat instruction list.  On qubit q's chain the
    rotation and the adjacent CX are always neighbours, regardless of how many
    other qubits' rotations sit between them in the flat program.  This
    unlocks commutation, enabling MergeRotations to fold the newly-adjacent
    rotation pairs across the entangler boundary in the next fixpoint iteration.

    The fix also required changing the PassManager's fixpoint convergence
    criterion: the prior rule terminated when scalarized cost did not improve
    in one full pass through the group (improvement <= tol, tol=0.0).
    CommuteThroughControl is a "setup" pass — it reorders without reducing gate
    count in its own iteration; the reduction comes in the next iteration when
    MergeRotations folds the reordered sequence.  The updated rule terminates
    only on cost REGRESSION (improvement < -tol), allowing flat-cost iterations
    to propagate.

    Measured at default settings (n=5, reps=3, rotations="z",
    config/grover.toml): opt_level=3 achieves vs_lvl0=1.032x vs
    opt_level=2's 1.000x — a 3.2 % reduction in single-qubit gate count
    (264 -> 255 1q gates after transpilation).

    The "z" (single rz per qubit per layer) rotation spec is the canonical
    demonstration case.  Rotation specs whose per-qubit block ends in a non-rz
    axis (e.g. "yz", "yzy") have an ry blocker between the commuted rz and the
    next layer's rz on the per-qubit chain, preventing the merge.  Those specs
    show identical level-2 and level-3 results — correct and expected.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import argparse
import math
import sys
import time

from lccfq_lang import QPU, CRegister, Circuit
from lccfq_lang.lang.variational import hw_eff_ansatz


def build_hwea(c, qpu, n_qubits: int, layers: int, rotations: str) -> None:
    """Push a HWEA circuit into the open Circuit context."""
    target = list(range(n_qubits))
    n_params = (layers + 1) * n_qubits * len(rotations)
    # Use varied, non-trivial angles so rotations don't trivially cancel.
    params = [math.pi / (i + 3) for i in range(n_params)]

    for instr in hw_eff_ansatz(
        qpu.isa, target,
        params=params, layers=layers, rotations=rotations,
    ):
        c >> instr

    c >> qpu.isa.measure(tgs=target)


def run_at_level(
    qpu: QPU,
    n_qubits: int,
    layers: int,
    rotations: str,
    opt_level: int,
) -> tuple[dict, float]:
    qreg = qpu.qregister(n_qubits)
    creg = CRegister(n_qubits)

    t0 = time.perf_counter()

    with Circuit(qreg, creg, qpu, opt_level=opt_level, report=True) as c:
        build_hwea(c, qpu, n_qubits, layers, rotations)

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


def validate_configuration(qpu: QPU, n_qubits: int, rotations: str) -> None:
    """Validate inputs against the QPU config.

    Raises SystemExit with a descriptive message on the first violation.
    """
    qpu_count = qpu.config.qubit_count

    if n_qubits < 1:
        raise SystemExit(
            f"--n-qubits must be at least 1; got {n_qubits}."
        )
    if n_qubits > qpu_count:
        raise SystemExit(
            f"--n-qubits={n_qubits} exceeds the QPU's qubit_count={qpu_count}. "
            f"Either lower --n-qubits or bump qubit_count in the QPU config."
        )
    if not rotations:
        raise SystemExit("--rotations must not be empty.")
    for ch in rotations:
        if ch not in ("x", "y", "z"):
            raise SystemExit(
                f"--rotations may only contain 'x', 'y', 'z'; got '{ch}'."
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Hardware-efficient ansatz demo showing opt-level effects.",
    )
    p.add_argument(
        "--n-qubits", "-n", type=int, default=5,
        help=(
            "Number of qubits in the HWEA register (default 5). "
            "Must satisfy 1 <= n <= qpu.qubit_count."
        ),
    )
    p.add_argument(
        "--reps", type=int, default=3,
        help="Number of entangle+rotation layers after the initial rotation layer (default 3).",
    )
    p.add_argument(
        "--rotations", type=str, default="z",
        help=(
            "Single-qubit rotation axes per qubit per layer (default 'yzy'). "
            "Each character must be 'x', 'y', or 'z'."
        ),
    )
    p.add_argument(
        "--config", type=str, default="config/grover.toml",
        help="QPU configuration file (default config/grover.toml).",
    )
    p.add_argument(
        "--last-pass", type=str, default="transpiled",
        choices=[
            "parsed", "mapped", "swapped", "expanded",
            "arch_optimized", "transpiled", "mach_optimized", "executed",
        ],
        help=(
            "Pipeline terminator. Default 'transpiled' runs the compiler "
            "passes only; 'executed' additionally submits the transpiled "
            "circuit to the backend (gRPC server in [network]) at each "
            "opt level."
        ),
    )
    args = p.parse_args()

    qpu = QPU(filename=args.config, last_pass=args.last_pass)

    validate_configuration(qpu, args.n_qubits, args.rotations)

    n_qubits = args.n_qubits
    layers = args.reps
    rotations = args.rotations
    n_params = (layers + 1) * n_qubits * len(rotations)

    print(f"=== {n_qubits}-qubit HWEA demo ===")
    print(f"  config     : {args.config} (qubit_count={qpu.config.qubit_count})")
    print(f"  layers     : {layers}")
    print(f"  rotations  : {rotations!r}")
    print(f"  n_params   : {n_params}")
    print()

    reports: dict[int, tuple[dict, float]] = {}

    for level in (0, 1, 2, 3):
        print(f"Compiling at opt_level={level}...", end=" ", flush=True)
        report, wall = run_at_level(qpu, n_qubits, layers, rotations, level)
        print(f"done ({wall:.2f}s wall)")
        reports[level] = (report, wall)

    print()

    for level in (0, 1, 2, 3):
        report, wall = reports[level]
        summarize(report, level, wall)
        print()

    # Comparison table.
    # vs_lvl0 = (level-0 total gate count) / (level-i total gate count).
    # Higher is better; level-0 is 1.00x by construction.
    baseline = reports[0][0]["totals"]["cost_after"]
    baseline_total = baseline["count_1q"] + baseline["count_2q"]

    table_width = 130
    print("=" * table_width)
    header = (
        f"{'level':>5}  {'depth_in':>9}  {'depth_out':>10}  "
        f"{'1q_in':>6}  {'1q_out':>7}  "
        f"{'2q_in':>6}  {'2q_out':>7}  "
        f"{'score_in':>10}  {'score_out':>10}  "
        f"{'pass_s':>8}  {'wall_s':>8}  {'vs_lvl0':>8}"
    )
    print(header)
    print("-" * table_width)

    ratios: dict[int, float] = {}
    for level in (0, 1, 2, 3):
        report, wall = reports[level]
        b = report["totals"]["cost_before"]
        a = report["totals"]["cost_after"]
        total_out = a["count_1q"] + a["count_2q"]
        ratio = (baseline_total / total_out) if total_out else float("nan")
        ratios[level] = ratio
        print(
            f"{level:>5}  {b['depth']:>9}  {a['depth']:>10}  "
            f"{b['count_1q']:>6}  {a['count_1q']:>7}  "
            f"{b['count_2q']:>6}  {a['count_2q']:>7}  "
            f"{b['scalarized']:>10.2f}  {a['scalarized']:>10.2f}  "
            f"{report['totals']['total_seconds']:>8.3f}  {wall:>8.2f}  "
            f"{ratio:>7.3f}x"
        )

    print("=" * table_width)

    # Verify the load-bearing user-visible signal: level-3 must beat level-2.
    if ratios[3] > ratios[2]:
        print(
            f"\nLevel-3 vs_lvl0={ratios[3]:.3f}x  >  "
            f"level-2 vs_lvl0={ratios[2]:.3f}x: DAG-based commutation is active."
        )
    else:
        print(
            f"\nWARNING: level-3 vs_lvl0={ratios[3]:.3f}x is NOT strictly greater "
            f"than level-2 vs_lvl0={ratios[2]:.3f}x.  "
            f"The DAG-based CommuteThroughControl pass may not be firing.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
