# lccfq-lang

`lccfq-lang` is a research compiler for quantum circuits implemented in Python. Programs are written using a Python API and lowered through a six-stage compilation pipeline to the XYiSW native gate set targeted at superconducting hardware within the Leadership Class Compute Facility Project (LCCF). A backend layer is responsible for execution; at present, that backend layer is a stub.

> **Backend status.** `lccfq-lang` performs compilation only. The method `QPU.exec_circuit` does not currently communicate with physical hardware. All compilation behavior — gate decomposition, routing, optimization, and transpilation — is fully implemented and tested. Execution semantics are not. References to "running" a circuit mean lowering it to the point at which hardware would execute it. See [`wiki/13-limitations-roadmap.md`](wiki/13-limitations-roadmap.md) for details.

---

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) (package manager)

No additional system packages are required. Runtime dependencies (`networkx`, `numpy`, `toml`) and the development dependency (`pytest`) are installed automatically by `uv sync`.

---

## Installation

```bash
git clone <repository-url>
cd lccfq-lang
uv sync
```

`uv sync` creates a virtual environment under `.venv/` and installs all declared dependencies.

---

## Quick start

The following program, taken verbatim from `examples/bell_state.py`, prepares a two-qubit Bell state.

```python
# examples/bell_state.py
from lccfq_lang import QPU, CRegister, Circuit

def bell_state():
    qpu = QPU(filename="config/default.toml")
    qreg = qpu.qregister(2)
    creg = CRegister(size=2)

    with Circuit(qreg, creg, qpu, shots=1000) as c:
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.cx(ct=0, tg=1)
        c >> qpu.isa.measure(tgs=[0, 1])

    print(creg.frequencies())

if __name__ == "__main__":
    bell_state()
```

Instructions are appended to a circuit with the `>>` operator. When the `with` block exits, the circuit is compiled through the pipeline up to the stage specified by `last_pass` (default: all six stages). On a functional backend the expected result is approximately 50 percent `|00⟩` and 50 percent `|11⟩`; with the current stub the returned dictionary is empty.

Run the example:

```bash
uv run python examples/bell_state.py
```

---

## Compilation pipeline

The compiler transforms an input program through six ordered stages. Each stage can be used as a `last_pass` checkpoint.

| Stage | `last_pass` value | Function |
|---|---|---|
| 1 | `mapped` | Virtual-to-physical qubit assignment |
| 2 | `swapped` | SWAP insertion to satisfy hardware adjacency constraints |
| 3 | `expanded` | Decomposition of compound gates (u2, u3, controlled-U, multi-controlled) into elementary primitives |
| 4 | `arch_optimized` | Ideal-gate optimization (peephole, commutation, template rewrites) — active when `opt_level > 0` |
| 5 | `transpiled` | Translation into the XYiSW native gate set (rx, ry, sqiSWAP, measure) |
| 6 | `mach_optimized` | Native-gate optimization (rotation fusion, redundant-sandwich elision) — active when `opt_level > 0` |

Stopping the pipeline at an intermediate stage:

```python
qpu = QPU(filename="config/default.toml", last_pass="transpiled")
```

---

## Key features

- **Six-stage compilation pipeline** from virtual gates to native hardware sequences.
- **XYiSW native gate set** — `sqiSWAP`-based decompositions verified correct across `cx`, `cy`, `cz`, `ch`, `cp`, `cphase`, `crx`, `cry`, `crz`, and `swap`.
- **Optimization levels 0–3** via `opt_level` on the `Circuit` constructor; level 2 and above force the `sabre_fast` routing strategy.
- **Three routing strategies**: `identity` (trivial, for fully connected topologies), `sabre_lite` (lightweight SABRE), and `sabre_fast` (LightSABRE, arXiv:2409.08368).
- **Explicit pass control** via `opt_passes=[...]` for full optimization customization.
- **Pipeline telemetry** via `report=True` — per-pass gate counts, circuit depth, estimated error, and wall-clock time, returned as a JSON-serializable `opt_report` dict.
- **High-level block API** (`BlockFactory`) with 24 block types including multi-control gates (MCX, MCZ, MCRY, MCRZ) using polynomial-cost decomposition.
- **OpenQASM 3.0 export** via `QASMSynthesizer`.
- **Custom pass API** — subclass `Pass`, implement `run(program, ctx)` as a pure function, and register with `register_template`.
- **Hardware test context** (`Test`) for characterization primitives such as resonator spectroscopy and power Rabi.

---

## Optimization

The `opt_level` keyword on `Circuit` selects the optimization intensity:

| Level | Arch passes | Mach passes | Routing forced |
|------:|---|---|---|
| `0` | _(none)_ | _(none)_ | per QPU config |
| `1` | RemoveIdentity, CancelInverses, MergeRotations | MergeAdjacent1Q, RemoveIdentityMach | per QPU config |
| `2` | adds FuseEulerZYZ, HCXHRule, SwapElision | adds RyRzRyToHardware, DeferMeasurement | `sabre_fast` |
| `3` | adds CommuteThroughControl | adds EulerXYRecompose, ParallelizeLayers | `sabre_fast` |

Pass `report=True` to collect per-pass telemetry:

```python
with Circuit(qreg, creg, qpu, opt_level=2, report=True) as c:
    ...

print(c.opt_report["totals"])
# {"cost_before": {...}, "cost_after": {...},
#  "scalarized_delta": 12.3, "total_seconds": 0.004}
```

For full pass-level control, supply `opt_passes=[...]` with explicit pass names. When `opt_passes` is set, `opt_level` is ignored.

---

## Examples

All programs are in the `examples/` directory and can be run with `uv run python examples/<file>`.

| File | Description |
|---|---|
| `bell_state.py` | Two-qubit Bell-state preparation — minimal working example |
| `teleportation.py` | Quantum teleportation protocol with classical post-processing |
| `dj.py` | Deutsch–Jozsa algorithm on 3 qubits |
| `grover.py` | Grover's search algorithm (general) |
| `grover5.py` | 5-qubit Grover search with optimization-level comparison |
| `opt_demo.py` | Optimization pipeline demonstration across multiple `opt_level` settings |
| `xeb_rcs.py` | Cross-entropy benchmarking via random circuit sampling |

---

## Testing

```bash
uv run pytest -q
```

The suite collects 1,258 tests and completes in approximately 6 seconds. The suite is expected to be fully green on every commit to `main`. For verbose output:

```bash
uv run pytest -v --tb=short
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Contact: nunezco2@illinois.edu
