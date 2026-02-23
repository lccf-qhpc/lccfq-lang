"""
Filename: qasm.py
Author: Santiago Nunez-Corrales
Date: 2025-08-06
Version: 1.0
Description:
    This file implements the synthesizer to OpenQASM.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import os

from typing import List, Dict, Optional
from ..instruction import Instruction
from ..context import Circuit
from ..error import UnknownInstruction, MalformedInstruction


class QASMSynthesizer:
    """Handler for LCCFQ code to OpenQASM 3.0

    """

    def __init__(self):
        self.gate_map: Dict[str, str] = {
            # Single-qubit non-parametric
            "x": "x", "y": "y", "z": "z", "h": "h",
            "s": "s", "sdg": "sdg", "t": "t", "tdg": "tdg",

            # Single-qubit parametric
            "rx": "rx", "ry": "ry", "rz": "rz", "p": "p",
            "phase": "phase", "u2": "u2", "u3": "u3",

            # Two-qubit gates
            "cx": "cx", "cy": "cy", "cz": "cz", "ch": "ch",
            "cp": "cp", "crx": "crx", "cry": "cry",
            "crz": "crz", "cphase": "cphase", "cu": "cu",
            "swap": "swap",

            # Meta
            "measure": "measure",
            "reset": "reset"
        }

    def synth_circuit(self, circuit: Circuit, path: Optional[str] = None) -> str:
        """
        Convert a full Circuit into an OpenQASM 3.0 program.

        :param circuit: LCCFQ circuit object
        :return: string with valid QASM 3.0 code
        """
        lines = self.get_qasm_header(
            n_qubits=circuit.qreg.qubit_count,
            n_bits=circuit.creg.bit_count
        )

        for instr in circuit.instructions:
            qasm_line = self.synth_instruction(instr)
            lines.append(qasm_line)

        qasm_code = "\n".join(lines)

        if path:
            dirpath = os.path.dirname(path)

            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(path, "w") as f:
                f.write(qasm_code)

        return qasm_code

    def synth_instruction(self, instr: Instruction) -> str:
        """
        Synthesize a single instruction into OpenQASM 3.0.

        :param instr: instruction
        :return: OpenQASM code
        """
        symbol = instr.symbol
        qasm_op = self.gate_map.get(symbol)

        if qasm_op is None:
            raise UnknownInstruction(instr)

        tgs = [f"q[{q}]" for q in (instr.target_qubits or [])]
        cts = [f"q[{c}]" for c in (instr.control_qubits or [])]

        # OpenQASM 3.0: control(s) target(s)
        qubit_args = cts + tgs

        if instr.parameters:
            param_str = ", ".join(f"{p:.10g}" for p in instr.parameters) #OpenQASM 3.0: numeric precision convention
            gate_call = f"{qasm_op}({param_str})"
        else:
            gate_call = qasm_op

        # Measurement special case
        if symbol == "measure":
            if not tgs:
                raise MalformedInstruction(instr, "No target qubits")
            return "\n".join(
                f"measure {q} -> c[{i}];" for i, q in enumerate(tgs)
            )

        if symbol == "reset":
            return "\n".join(f"reset {q};" for q in tgs)

        return f"{gate_call} {' , '.join(qubit_args)};"

    @staticmethod
    def get_qasm_header(n_qubits: int, n_bits: int) -> List[str]:
        """
        Generate the header for an OpenQASM 3.0 program.

        :param n_qubits: number of qubits
        :param n_bits: number of classical bits
        :return: QASM header string
        """
        return [
            f"OPENQASM 3.0;",
            f"qubit[{n_qubits}] q;",
            f"bit[{n_bits}] c;"
        ]
