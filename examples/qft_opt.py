"""
Filename: qft_opt.py
Author: Santiago Nunez-Corrales
Date: 2026-05-21
Version: 1.0
Description:
    Parametric quantum Fourier transform demo. Showcases the effect of each
    optimization level (0, 1, 2, 3) on a textbook QFT circuit of arbitrary
    width.

    The QFT block uses only single-qubit Hadamards, controlled-phase
    rotations, and optional trailing bit-reversal SWAPs. No multi-control
    decomposition is required, so this example exposes a different mix of
    optimization opportunities than grover_opt.py: rotation merging,
    canonical rz collapse, and swap elision dominate here.

    The inverse QFT may be selected with --inverse.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import argparse
import sys
import time

from lccfq_lang import QPU, CRegister, Circuit
from lccfq_lang.lang.blocks import BlockFactory, BlockType


def build_qft(
    c: Circuit,
    qpu: QPU,
    factory: BlockFactory,
    n_qubits: int,
    inverse: bool,
    do_swaps: bool,
    endianness: str,
) -> None:
    """Push a QFT (or IQFT) circuit into the open Circuit context."""
    target = list(range(n_qubits))
    block_type = BlockType.IQFT if inverse else BlockType.QFT

    for instr in factory.block(
        block_type,
        target,
        do_swaps=do_swaps,
        endianness=endianness,
    ):
        c >> instr

    c >> qpu.isa.measure(tgs=target)


def run_at_level(
    qpu: QPU,
    n_qubits: int,
    opt_level: int,
    inverse: bool,
    do_swaps: bool,
    endianness: str,
) -> tuple[dict, float]:
    qreg = qpu.qregister(n_qubits)
    creg = CRegister(n_qubits)
    factory = BlockFactory(qreg, creg)

    t0 = time.perf_counter()

    with Circuit(qreg, creg, qpu, opt_level=opt_level, report=True) as c:
        build_qft(c, qpu, factory, n_qubits, inverse, do_swaps, endianness)
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


def validate_configuration(qpu: QPU, n_qubits: int) -> None:
    """Validate user inputs against the QPU config.

    Raises SystemExit with a descriptive message on the first violation,
    so the user sees a single clear error rather than a stack trace from
    deep inside the compiler.
    """
    qpu_count = qpu.config.qubit_count

    if n_qubits < 1:
        raise SystemExit(
            f"--n-qubits must be at least 1 (QFT acts on a non-empty register); "
            f"got {n_qubits}."
        )
    if n_qubits > qpu_count:
        raise SystemExit(
            f"--n-qubits={n_qubits} exceeds the QPU's qubit_count={qpu_count}. "
            f"Either lower --n-qubits or bump qubit_count in the QPU config."
        )
    if n_qubits == 1:
        print(
            "warning: --n-qubits=1 reduces QFT to a single Hadamard; the "
            "optimization deltas across levels will be uninteresting.",
            file=sys.stderr,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parametric QFT demo showing opt-level effects.",
    )
    p.add_argument(
        "--n-qubits", "-n", type=int, default=4,
        help=(
            "QFT register width in qubits (default 4). Must satisfy "
            "1 <= n <= qpu.qubit_count."
        ),
    )
    p.add_argument(
        "--config", type=str, default="config/default.toml",
        help="QPU configuration file (default config/default.toml).",
    )
    p.add_argument(
        "--inverse", action="store_true",
        help="Build the inverse QFT (IQFT) instead of the forward QFT.",
    )
    p.add_argument(
        "--no-swaps", action="store_true",
        help=(
            "Skip the trailing bit-reversal SWAPs. Useful when the QFT "
            "feeds directly into another bit-reversed consumer (e.g. the "
            "Draper adder)."
        ),
    )
    p.add_argument(
        "--endianness", choices=["little", "big"], default="little",
        help=(
            "Qubit-order convention for the QFT (default little). 'little' "
            "treats target[0] as the least-significant qubit."
        ),
    )
    args = p.parse_args()

    qpu = QPU(filename=args.config, last_pass="executed")

    validate_configuration(qpu, args.n_qubits)

    n_qubits = args.n_qubits
    do_swaps = not args.no_swaps
    label = "IQFT" if args.inverse else "QFT"

    print(f"=== {n_qubits}-qubit {label} demo ===")
    print(f"  config          : {args.config} (qubit_count={qpu.config.qubit_count})")
    print(f"  do_swaps        : {do_swaps}")
    print(f"  endianness      : {args.endianness}")
    print()

    reports: dict[int, tuple[dict, float]] = {}

    for level in (0, 1, 2, 3):
        print(f"Compiling at opt_level={level}...", end=" ", flush=True)
        report, wall = run_at_level(
            qpu, n_qubits, level, args.inverse, do_swaps, args.endianness,
        )
        print(f"done ({wall:.2f}s wall)")
        reports[level] = (report, wall)

    print()

    for level in (0, 1, 2, 3):
        report, wall = reports[level]
        summarize(report, level, wall)
        print()

    # Comparison table.
    # The vs_lvl0 column is the ratio of level 0's total output gates to
    # level i's, where "total" = 1q + 2q after the final optimization stage.
    # Higher is better; level 0 is 1.00 by construction.
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

    for level in (0, 1, 2, 3):
        report, wall = reports[level]
        b = report["totals"]["cost_before"]
        a = report["totals"]["cost_after"]
        total_out = a["count_1q"] + a["count_2q"]
        ratio = (baseline_total / total_out) if total_out else float("nan")
        print(
            f"{level:>5}  {b['depth']:>9}  {a['depth']:>10}  "
            f"{b['count_1q']:>6}  {a['count_1q']:>7}  "
            f"{b['count_2q']:>6}  {a['count_2q']:>7}  "
            f"{b['scalarized']:>10.2f}  {a['scalarized']:>10.2f}  "
            f"{report['totals']['total_seconds']:>8.3f}  {wall:>8.2f}  "
            f"{ratio:>7.3f}x"
        )

    print("=" * table_width)


if __name__ == "__main__":
    main()
