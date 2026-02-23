# lccfq-lang

A Python library for programming quantum processing units (QPUs) within the Lambda Calculus Compiler Framework (LCCF). It provides a high-level DSL for defining quantum circuits, a multi-stage compilation pipeline that maps virtual circuits to physical hardware, and OpenQASM 3.0 synthesis.

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
