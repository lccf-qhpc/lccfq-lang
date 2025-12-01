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
from .error import UnknownCompilerPass
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

    def _handle_pass(self, program: List[Instruction]|List[Gate], cpass: str) -> None:
        """Handle compiler pass termination

        :param program: list of entities in different stages of processing composing a program
        :param cpass: current compilation pass
        :return: nothing
        """
        if self.verbose:
            print(f"Stage: {cpass}")
            print(program)
            print("\n\n")

        if cpass == "executed":
            result = self.qreg.qpu.exec_circuit(program, self.shots)
        else:
            result = {format(i, f"0{self.creg.bit_count}b"): -1 for i in range(2 ** self.creg.bit_count)}

        self.creg.absorb(result)

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
        if self.qreg.qpu.last_pass == "parsed":
            self._handle_pass(self.instructions, self.qreg.qpu.last_pass)
            return True

        # Step 1: map all instructions from virtual qubits to physical qubits
        mapped =  list(
            map(lambda instr: self.qreg.map(instr), self.instructions)
        )

        if self.qreg.qpu.last_pass == "mapped":
            self._handle_pass(mapped, self.qreg.qpu.last_pass)
            return True

        # Step 2: introduce any required swaps (more swaps if done after expanding
        swapped = list(
            chain.from_iterable(
                map(lambda instr: self.qreg.swaps(instr, self.qreg.qpu.isa), self.instructions)
            )
        )

        if self.qreg.qpu.last_pass == "swapped":
            self._handle_pass(swapped, self.qreg.qpu.last_pass)
            return True

        # Step 3: expand special instructions into nicely realizable gates
        expanded = list(
            chain.from_iterable(
                map(lambda instr: self.qreg.expand(instr), swapped)
            )
        )

        if self.qreg.qpu.last_pass == "expanded":
            self._handle_pass(expanded, self.qreg.qpu.last_pass)
            return True

        # Step 4: transpile finally into Gates
        transpiled = list(
            chain.from_iterable(
                map(lambda instr: self.qreg.qpu.transpiler.transpile_gate(instr), expanded)
            )
        )

        if self.qreg.qpu.last_pass == "transpiled":
            self._handle_pass(transpiled, self.qreg.qpu.last_pass)
            return True

        if self.qreg.qpu.last_pass == "executed":
            self._handle_pass(transpiled, self.qreg.qpu.last_pass)
            return True
        else:
            raise UnknownCompilerPass(self.qreg.qpu.last_pass)


class Test:
    """Implementaiton of a test context for LCCFQ.

    A test context is either a specialized instruction corresponding to hardware-level
    primitives (e.g., power Rabi, resonator spectroscopy) or the use of a single quantum
    instruction as a test.

    """
    def __init__(self,
                 qreg: QRegister,
                 accum: Dict[int,Dict[str,float]],
                 verbose=False):
        """Create a new test.

        :param qreg: quantum register
        :param shots: number of shot to run this circuit for
        :param verbose: trigger verbose output
        """
        self.qreg = qreg
        self.accum = accum
        self.verbose = verbose
        self.instructions: List[Instruction] = list()

    def __rshift__(self, instr: Instruction):
        """Add a new instruction to the circuit using the `>>` operator. This
        removes verbosity.

        :param instr: instruction to add
        :return: none
        """

        # We try to catch errors as early as they appear, which is
        # when these are included in the code.
        self.qreg.challenge(instr, QContext.TEST)
        self.instructions.append(instr)

    def __enter__(self):
        """Enter the context

        :return: the circuit itself
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exiting the context leaves in the dictionary at the start
        the corresponding output per instruction type.

        :param exc_type: none
        :param exc_val: none
        :param exc_tb: none
        :return: nothing
        """
        for i, instruction in enumerate(self.instructions):
            res = self.qreg.qpu.exec_single(instruction, instruction.shots)
            self.accum[i] = res
