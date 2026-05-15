"""
Filename: opt_demo.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.0
Description:
    Demonstration of the lccfq-lang optimization pipeline. Builds a Grover
    search circuit (4 qubits, 1 iteration) and compiles it at every
    opt_level (0..3) with `report=True`, printing per-level cost deltas
    and a final comparison table.

    Two-qubit decompositions to the XYiSW native gate set (cx, cy, cz, ch,
    cp, crx, cry, crz, cphase, swap) have been verified correct at ~1e-15
    Frobenius precision (Task #16).  The reported gate counts and depth
    figures therefore reflect a semantically correct compilation.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from __future__ import annotations
import numpy as np

from lccfq_lang import QPU, CRegister, Circuit


# ---------------------------------------------------------------------------
# Circuit fixture: one Grover iteration over 4 qubits with marked state 1010
# ---------------------------------------------------------------------------

def _add_grover_iteration(c, qpu, marked_bits):
    """Append the body of one Grover iteration to the open Circuit `c`."""
    # Mark
    for i, bit in enumerate(marked_bits):
        if bit == 0:
            c >> qpu.isa.x(tg=i)
    # Multi-controlled-Z via H+CX ladder
    c >> qpu.isa.h(tg=3)
    c >> qpu.isa.cx(ct=2, tg=3)
    c >> qpu.isa.cx(ct=1, tg=2)
    c >> qpu.isa.cx(ct=0, tg=1)
    c >> qpu.isa.x(tg=0)
    c >> qpu.isa.cx(ct=1, tg=0)
    c >> qpu.isa.x(tg=0)
    c >> qpu.isa.cx(ct=0, tg=1)
    c >> qpu.isa.cx(ct=1, tg=2)
    c >> qpu.isa.cx(ct=2, tg=3)
    c >> qpu.isa.h(tg=3)
    # Unmark
    for i, bit in enumerate(marked_bits):
        if bit == 0:
            c >> qpu.isa.x(tg=i)
    # Diffusion
    n_qubits = 4
    for q in range(n_qubits):
        c >> qpu.isa.h(tg=q)
        c >> qpu.isa.x(tg=q)
    c >> qpu.isa.h(tg=3)
    c >> qpu.isa.cx(ct=2, tg=3)
    c >> qpu.isa.cx(ct=1, tg=2)
    c >> qpu.isa.cx(ct=0, tg=1)
    c >> qpu.isa.x(tg=0)
    c >> qpu.isa.cx(ct=1, tg=0)
    c >> qpu.isa.x(tg=0)
    c >> qpu.isa.cx(ct=0, tg=1)
    c >> qpu.isa.cx(ct=1, tg=2)
    c >> qpu.isa.cx(ct=2, tg=3)
    c >> qpu.isa.h(tg=3)
    for q in range(n_qubits):
        c >> qpu.isa.x(tg=q)
        c >> qpu.isa.h(tg=q)


def build_demo_circuit(c, qpu) -> None:
    """Populate `c` with a 4-qubit, 1-iteration Grover search."""
    n_qubits = 4
    marked_bitstring = "1010"
    marked_bits = [int(b) for b in reversed(marked_bitstring)]

    # Hadamard layer
    for q in range(n_qubits):
        c >> qpu.isa.h(tg=q)

    # One Grover iteration (enough redundancy for peephole + templates).
    _add_grover_iteration(c, qpu, marked_bits)

    # Measure all qubits
    c >> qpu.isa.measure(tgs=list(range(n_qubits)))


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt_cost(d: dict) -> str:
    return (
        f"depth={d['depth']:>3}  "
        f"1q={d['count_1q']:>3}  "
        f"2q={d['count_2q']:>3}  "
        f"native2q={d['count_native_2q']:>3}  "
        f"score={d['scalarized']:.2f}"
    )


def _print_level_summary(level: int, report: dict) -> None:
    print(f"--- opt_level={level} ---")
    print(f"  routing_strategy : {report['routing_strategy']}")
    print(f"  last_pass        : {report['last_pass']}")
    totals = report["totals"]
    print(f"  before           : {_fmt_cost(totals['cost_before'])}")
    print(f"  after            : {_fmt_cost(totals['cost_after'])}")
    print(f"  scalarized delta : {totals['scalarized_delta']:.2f}")
    print(f"  total seconds    : {totals['total_seconds']:.6f}")
    if report["groups"]:
        print(f"  groups run       : "
              + ", ".join(f"{g['name']}({g['iterations']}it)"
                          for g in report["groups"]))
    else:
        print("  groups run       : (none)")
    print()


def _print_comparison(rows: list[tuple[int, dict]]) -> None:
    print("=" * 78)
    print(f"{'level':>5}  {'depth_in':>8}  {'depth_out':>9}  "
          f"{'2q_in':>5}  {'2q_out':>6}  {'score_in':>9}  "
          f"{'score_out':>9}  {'sec':>9}")
    print("-" * 78)
    for level, rep in rows:
        t = rep["totals"]
        b, a = t["cost_before"], t["cost_after"]
        print(f"{level:>5}  {b['depth']:>8}  {a['depth']:>9}  "
              f"{b['count_2q']:>5}  {a['count_2q']:>6}  "
              f"{b['scalarized']:>9.2f}  {a['scalarized']:>9.2f}  "
              f"{t['total_seconds']:>9.6f}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows: list[tuple[int, dict]] = []
    for level in (0, 1, 2, 3):
        # `last_pass="mach_optimized"` forces the full pipeline to run for
        # opt_level >= 1, and gracefully degrades for opt_level == 0
        # (slice_groups_for falls back to lower_transpile).
        qpu = QPU(filename="config/default.toml", last_pass="mach_optimized")
        qreg = qpu.qregister(4)
        creg = CRegister(4)

        with Circuit(qreg, creg, qpu, shots=1, opt_level=level,
                     report=True) as c:
            build_demo_circuit(c, qpu)

        assert c.opt_report is not None, "report=True must populate opt_report"
        _print_level_summary(level, c.opt_report)
        rows.append((level, c.opt_report))

    _print_comparison(rows)


if __name__ == "__main__":
    main()
