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
from dataclasses import dataclass
from .instruction import Instruction
from .register import QRegister, CRegister, QContext
from .error import UnknownCompilerPass
from .protocol import Backend
from typing import List, Dict, Callable, Tuple


@dataclass
class CompilerPass:
    """A single named stage in the compilation pipeline."""
    name: str
    transform: Callable[[list], list]


class CompilationPipeline:
    """An ordered sequence of compiler passes."""

    def __init__(self, passes: List[CompilerPass]):
        self.passes = passes

    def run(self, instructions: List[Instruction], last_pass: str) -> Tuple[str, list]:
        """Run passes up to and including last_pass.

        :param instructions: raw instruction list
        :param last_pass: name of the final pass to execute
        :return: (pass_name, result) of the last pass that ran
        """
        program = instructions

        for cpass in self.passes:
            program = cpass.transform(program)

            if cpass.name == last_pass:
                return cpass.name, program

        raise UnknownCompilerPass(last_pass)


class Circuit:
    """Implementation of a quantum circuit based on instructions.

    A circuit is an atomic program composed of gates and measurements at the
    end. Circuits, after their context closes, generate code that goes into
    a circuit description to the respective backend.
    """

    def __init__(self,
                 qreg: QRegister,
                 creg: CRegister,
                 qpu: Backend = None,
                 shots: int = 1000,
                 verbose: bool = False,
                 opt_level: int = 0,
                 opt_passes: list[str] | None = None):
        """Create a new circuit.

        :param qreg: quantum register
        :param creg: classical register
        :param qpu: QPU backend
        :param shots: number of shots
        :param verbose: trigger verbose output
        :param opt_level: arch-level optimization level (0..3). Ignored
            when opt_passes is not None. Default 0 (no arch_opt).
        :param opt_passes: explicit list of arch pass names; overrides opt_level.
            Use [] to explicitly disable arch_opt while still using the
            explicit-mode contract.
        """
        if not isinstance(opt_level, int) or opt_level not in (0, 1, 2, 3):
            raise ValueError(
                f"Circuit: opt_level must be one of (0, 1, 2, 3), got {opt_level!r}"
            )
        if opt_passes is not None:
            if not isinstance(opt_passes, list) or not all(
                isinstance(n, str) for n in opt_passes
            ):
                raise TypeError("Circuit: opt_passes must be None or list[str]")

        self.qreg = qreg
        self.creg = creg
        self.qpu = qpu
        self.shots = shots
        self.verbose = verbose
        self.instructions: List[Instruction] = list()
        self._opt_records = []
        self._opt_level = opt_level
        self._opt_passes = opt_passes

    def results(self) -> Dict[str, int]:
        return self.creg.data

    def frequencies(self):
        return self.creg.frequencies()

    def _handle_pass(self, program: list, cpass: str) -> None:
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
            result = self.qpu.exec_circuit(program, self.shots)
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
        challenged = self.qreg.challenge(instr, QContext.CIRCUIT)
        self.instructions.append(challenged)

    def __enter__(self):
        """Enter the context

        :return: the circuit itself
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, equivalent to generating the circuit and sending it to the
        backend. This allows multiple circuits to have multiple backends by construction.

        :param exc_type: exception type, if any
        :param exc_val: exception value, if any
        :param exc_tb: exception traceback, if any
        :return: False to propagate exceptions, True on success
        """
        if exc_type is not None:
            return False

        last_pass = self.qpu.last_pass

        if last_pass == "parsed":
            self._handle_pass(self.instructions, "parsed")
            return True

        # Deferred imports: keep arch/ free of eager opt/ dependency at module load.
        from lccfq_lang.opt import PassContext, PassManager
        from lccfq_lang.opt.builtin.lower_passes import (
            build_lowering_groups, slice_groups_for,
        )

        groups = slice_groups_for(
            last_pass,
            build_lowering_groups(
                self.qreg,
                self.qpu,
                opt_level=self._opt_level,
                opt_passes=self._opt_passes,
            ),
        )
        ctx = PassContext(
            qpu_config=self.qpu.config,
            isa=self.qpu.isa,
            mapping=self.qpu.mapping,
            topology=self.qpu.mapping.topology,
        )
        program, records = PassManager(groups).run(self.instructions, ctx)
        self._opt_records = records
        self._handle_pass(program, last_pass)
        return True


class Test:
    """Implementaiton of a test context for LCCFQ.

    A test context is either a specialized instruction corresponding to hardware-level
    primitives (e.g., power Rabi, resonator spectroscopy) or the use of a single quantum
    instruction as a test.

    """
    def __init__(self,
                 qreg: QRegister,
                 accum: Dict[int,Dict[str,float]],
                 qpu: Backend = None,
                 verbose=False):
        """Create a new test.

        :param qreg: quantum register
        :param accum: accumulator for test results
        :param qpu: QPU backend
        :param verbose: trigger verbose output
        """
        self.qreg = qreg
        self.accum = accum
        self.qpu = qpu
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
        challenged = self.qreg.challenge(instr, QContext.TEST)
        self.instructions.append(challenged)

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
            res = self.qpu.exec_single(instruction, instruction.shots)
            self.accum[i] = res

class Control:
    # TODO: control here means a change to the QPU state or its defining thresholds.
    pass