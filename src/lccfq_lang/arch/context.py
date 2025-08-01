"""
Filename: context.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides the definition for a quantum circuit as a sequence
    of gates on a number of qubits. Circuits utilize a number of qubits
    with no assumption about the mapping, which is provided by the machine
    model (`mach`).

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from .instruction import Instruction
from .register import QRegister, CRegister, QContext
from ..mach.ir import Gate
from typing import List, Dict
from itertools import chain


class Circuit:
    """Implementation of a quantum circuit based on instructions.

    A circuit is an atomic program composed of gates and measurements at the
    end. Circuits, after their context closes, generate code that goes into
    a circuit description to the respective backend.
    """

    def __init__(self,
                 qreg: QRegister,
                 creg: CRegister,
                 shots: int = 1000,
                 verbose=False):
        """Create a new circuit.

        :param qreg: quantum register
        :param creg: classical register
        :param shots: number of shot to run this circuit for
        :param verbose: trigger verbose output
        """
        self.qreg = qreg
        self.creg = creg
        self.shots = shots
        self.verbose = verbose
        self.instructions: List[Instruction] = list()

    def results(self) -> Dict[str, int]:
        return self.creg.data

    def frequencies(self):
        return self.creg.frequencies()

    def __rshift__(self, instr: Instruction):
        """Add a new instruction to the circuit using the `>>` operator. This
        removes verbosity.

        :param instr: instruction to add
        :return: none
        """

        # We try to catch errors as early as they appear, which is
        # when these are included in the code.
        self.qreg.challenge(instr, QContext.CIRCUIT)
        self.instructions.append(instr)

    def __enter__(self):
        """Enter the context

        :return: the circuit itself
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, equivalent to generating the circuit and sending it to the
        backend. This allows multiple circuits to have multiple backends by construction.

        :param exc_type: none
        :param exc_val: none
        :param exc_tb: none
        :return: none
        """
        # A dry run only prints the instructions and return all results in -1
        if self.qreg.qpu.last_pass == "dryrun":


            if self.verbose:
                print("Parsed instructions:")
                print(self.instructions)
                print("\n\n")

            return True

        # Step 1: map all instructions from virtual qubits to physical qubits
        mapped =  list(
            map(lambda instr: self.qreg.map(instr), self.instructions)
        )

        # Step 2: introduce any required swaps (more swaps if done after expanding
        swapped = list(
            chain.from_iterable(
                map(lambda instr: self.qreg.swaps(instr, self.qreg.qpu.isa), self.instructions)
            )
        )

        # Step 3: expand special instructions into nicely realizable gates

        expanded = list(
            chain.from_iterable(
                map(lambda instr: self.qreg.expand(instr), swapped)
            )
        )

        # Step 4: transpile finally into Gates
        transpiled = list(
            chain.from_iterable(
                map(lambda instr: self.qreg.qpu.transpiler.transpile_gate(instr), expanded)
            )
        )

        if self.qreg.qpu.last_pass == "executed":
            result = self.qreg.qpu.exec_circuit(transpiled)
        else:
            result = {format(i, f"0{self.creg.bit_count}b"): -1 for i in range(2 ** self.creg.bit_count)}

        if self.verbose:
            stage_outputs = {
                "mapped": ("Mapped instructions:", mapped),
                "swaps": ("Swapped instructions:", swapped),
                "expanded": ("Expanded instructions:", expanded),
                "transpiled": ("Transpiled gates:", transpiled),
                "executed": ("Execution result:", result),
            }

            label, data = stage_outputs.get(self.qreg.qpu.last_pass, ("Unknown stage", None))

            if data is not None:
                print(label)
                print(data)
                print("\n\n")

        self.creg.absorb(result)

        return True