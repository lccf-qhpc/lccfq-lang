# lccfq-lang

A Python library for programming quantum processing units (QPUs) within the Leadership Class Compute Facility Project (LCCF). It provides a high-level DSL for defining quantum circuits, a multi-stage compilation pipeline that maps virtual circuits to physical hardware, and OpenQASM 3.0 synthesis.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (package manager)

## Installation

```bash
git clone <repository-url>
cd lccfq-lang
uv sync
```

## Quick start

```python
from lccfq_lang import QPU, QRegister, CRegister, Circuit, ISA

# Connect to a QPU using a TOML configuration
qpu = QPU(filename="config/default.toml", last_pass="transpiled")
qreg = QRegister(2, qpu)
creg = CRegister(size=2)
isa = ISA("lccf")

# Define a Bell state circuit
with Circuit(qreg, creg, shots=1000) as c:
    c >> isa.h(tg=0)
    c >> isa.cx(ct=0, tg=1)
    c >> isa.measure(tgs=[0, 1])

print(creg.frequencies())
```

Instructions are appended to a circuit using the `>>` operator. When the `with` block exits, the circuit is automatically compiled through the pipeline up to the stage specified by `last_pass`.

## Compilation pipeline

The compiler runs up to six stages, controlled by `last_pass`:

| Stage | Description |
|---|---|
| `parsed` | Validates instructions and returns them as-is |
| `mapped` | Assigns virtual qubits to physical qubits |
| `swapped` | Inserts SWAP gates for non-adjacent two-qubit operations |
| `expanded` | Decomposes high-level gates (U2, U3, CU) into primitives |
| `transpiled` | Converts to native hardware gate set |
| `executed` | Submits to the QPU backend |

## Optimization

The compiler ships an optional pipeline of architecture-level (`arch_opt`) and
machine-level (`mach_opt`) optimization passes. They run as additional stages
between the lowering steps shown above and are gated by an `opt_level`
keyword on `Circuit`.

### Optimization levels

| Level | Arch passes | Mach passes | Routing |
|------:|---|---|---|
| `0` | _(none)_ | _(none)_ | mapping default |
| `1` | RemoveIdentity, CancelInverses, MergeRotations | MergeAdjacent1Q, RemoveIdentityMach | mapping default |
| `2` | adds FuseEulerZYZ, HCXHRule, SwapElision | adds RyRzRyToHardware, DeferMeasurement | forces `sabre_lite` |
| `3` | adds CommuteThroughControl | adds EulerXYRecompose, ParallelizeLayers | forces `sabre_lite` |

```python
with Circuit(qreg, creg, qpu, shots=1000, opt_level=2) as c:
    c >> qpu.isa.h(tg=0)
    c >> qpu.isa.cx(ct=0, tg=1)
    c >> qpu.isa.measure(tgs=[0, 1])
```

For full control, pass `opt_passes=[...]` with explicit pass names (resolved against
`ALL_ARCH_PASSES` / `ALL_MACH_PASSES`). When `opt_passes` is set, `opt_level` is ignored
and user-registered templates are **not** auto-appended.

### Inspecting the pipeline

`last_pass` accepts the lowering stages plus two optimization checkpoints:

| `last_pass` value | What is returned |
|---|---|
| `parsed` | raw instructions |
| `mapped` | after virtual→physical mapping |
| `swapped` | after SWAP insertion |
| `expanded` | after high-level decomposition |
| `arch_optimized` | after the `arch_opt` PassGroup (if `opt_level > 0`) |
| `transpiled` | after lowering to native gates |
| `mach_optimized` | after the `mach_opt` PassGroup (if `opt_level > 0`) |

If `opt_level == 0`, requesting `arch_optimized` or `mach_optimized` is silently
satisfied by the immediately preceding lowering stage (no error).

Pass `report=True` to attach a structured pipeline report to the circuit:

```python
with Circuit(qreg, creg, qpu, opt_level=2, report=True) as c:
    ...

print(c.opt_report["totals"])
# {"cost_before": {...}, "cost_after": {...},
#  "scalarized_delta": 12.3, "total_seconds": 0.004}

for group in c.opt_report["groups"]:
    print(group["name"], group["iterations"], group["scalarized_delta"])
```

The report dict has the shape:

```text
opt_report
├── opt_level: int
├── opt_passes: list[str] | None
├── routing_strategy: str
├── last_pass: str
├── groups: list of
│     ├── name: str               (e.g. "arch_opt", "mach_opt")
│     ├── mode: "linear" | "fixpoint"
│     ├── iterations: int
│     ├── passes: list of
│     │     ├── name: str
│     │     ├── iteration: int
│     │     ├── cost_before: dict   (depth, count_1q, count_2q,
│     │     ├── cost_after:  dict    count_native_2q, estimated_error,
│     │     └── delta_seconds: float scalarized)
│     ├── cost_before: dict
│     ├── cost_after: dict
│     └── scalarized_delta: float
└── totals
      ├── cost_before: dict
      ├── cost_after: dict
      ├── scalarized_delta: float
      └── total_seconds: float
```

The dict is JSON-serializable (all leaves are primitives or `None`).

### Custom passes

`Pass` is the abstract base for a single optimization pass. Subclass it,
implement `run(program, ctx) -> program` as a pure function (do not mutate
`program`), and register it via `register_template` so it is appended to
`arch_opt` whenever `opt_level >= 1`:

```python
from lccfq_lang import Pass, PassContext, register_template

class DropAllX(Pass):
    name = "drop_all_x"
    applies_to = "arch"

    def __init__(self, isa):
        self._isa = isa

    def run(self, program, ctx: PassContext):
        return [op for op in program
                if getattr(op.code, "value", None) != "x"]

register_template(DropAllX)

with Circuit(qreg, creg, qpu, opt_level=1, report=True) as c:
    c >> qpu.isa.x(tg=0)
    c >> qpu.isa.measure(tgs=[0])

print([g["name"] for g in c.opt_report["groups"]])
```

For mach-level passes (`applies_to = "mach"`), prefer the explicit-mode
contract: pass the class name in `opt_passes=[...]` rather than registering
as a template (templates are arch-only by design).

### Known limitations

- **Two-qubit transpilation (Task #16):** the current XYiSW native lowering
  does not always emit canonical-CNOT-equivalent unitaries. Optimization
  passes themselves preserve semantics, but the *transpiled* program a
  level `>=1` pipeline produces should not yet be trusted as correct on
  hardware. Compile-time metrics (depth, counts, timings) are reliable.

## Instruction set

The ISA provides the following operations:

**Single-qubit gates:** `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`

**Parametric single-qubit gates:** `rx`, `ry`, `rz`, `p`, `phase`, `u2`, `u3`

**Two-qubit gates:** `cx`, `cy`, `cz`, `ch`, `swap`

**Parametric two-qubit gates:** `cp`, `crx`, `cry`, `crz`, `cphase`, `cu`

**Meta:** `measure`, `reset`, `nop`

**Hardware tests:** `resfreq`, `satspect`, `powrab`, `pispec`, `resspect`, `dispshift`, `rocalib`

Gate construction uses keyword arguments:
- `tg` -- target qubit index (single-target gates)
- `tgs` -- target qubit list (multi-target operations like `measure`)
- `ct` -- control qubit index
- `params` -- parameter list for parametric gates
- `shots` -- per-instruction shot count (required in `Test` contexts)

## OpenQASM 3.0 synthesis

Export a circuit to OpenQASM 3.0:

```python
from lccfq_lang import QASMSynthesizer

synth = QASMSynthesizer()
qasm_code = synth.synth_circuit(circuit=c, path="output.qasm")
```

This writes the file and returns the QASM source as a string.

## Hardware tests

The `Test` context runs hardware characterization primitives:

```python
from lccfq_lang import Test

tests = {}

with Test(qreg, tests) as t:
    t >> isa.resfreq(tgs=[0], params=[10.0, 3.42, 3.44, 0.01], shots=300)
    t >> isa.powrab(tgs=[0], params=[-0.4, 0.4, 20.0e-3], shots=300)
```

Results are collected in the `tests` dictionary, keyed by instruction index.

## Examples

The `examples/` directory contains complete programs:

| File | Description |
|---|---|
| `bell_state.py` | Bell state entanglement |
| `teleportation.py` | Quantum teleportation with classical post-processing |
| `dj.py` | Deutsch-Jozsa algorithm |
| `grover.py` | Grover's search algorithm |
| `xeb_rcs.py` | Cross-entropy benchmarking via random circuit sampling |

## Running tests

```bash
uv run pytest
```

## Project structure

```
src/lccfq_lang/
    __init__.py          Public API
    backend.py           QPU interface and compilation driver
    defaults.py          Default configuration
    arch/                Architecture layer
        isa.py           Instruction set definition
        register.py      Quantum and classical registers
        context.py       Circuit and Test contexts
        instruction.py   Instruction representation
        mapping.py       Virtual-to-physical qubit mapping
        synth/qasm.py    OpenQASM 3.0 synthesizer
    mach/                Machine model layer
        topology.py      Physical qubit topology
        transpilers.py   Transpiler base class
        ir.py            Native gate IR
        sets/            Native gate set implementations
    sys/                 System configuration
        base.py          QPU config and connection
        factories/       Transpiler factory
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
