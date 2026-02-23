"""
Filename: isa.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines the instruction set architecture of LCCFQ. For reference against
    OpenQASM see:

        https://openqasm.com/language/standard_library.html#standard-library.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from typing import List
from .instruction import Instruction, InstructionType


def sq_nopar_gates(gate_names):
    """
    Make single qubit non-parametric gate methods.
    :param gate_names: strings with single gate names
    :return: decorator for target class
    """
    def decorator(cls):
        for name in gate_names:
            def mk_sg_method(gate_name):
                def sg_method(self, tg: int = 0, shots=None) -> Instruction:
                    return Instruction(
                        symbol=gate_name,
                        modifies_state=False,
                        is_controlled=False,
                        target_qubits=[tg],
                        control_qubits=None,
                        params=None,
                        shots=shots
                    )

                sg_method.__name__ = gate_name
                return sg_method

            setattr(cls, name, mk_sg_method(name))
        return cls

    return decorator

def sq_par_gates(gate_names):
    """
    Make single qubit parametric gate methods. Note we are backward compatible with
    OpenQASM 2 (`u2`, `u3)

    :param gate_names: strings with single gate names
    :return: decorator for target class
    """
    def decorator(cls):
        for name in gate_names:
            def mk_sg_method(gate_name):
                def sg_method(self, tg: int = 0, params=None, shots=None) -> Instruction:
                    return Instruction(
                        symbol=gate_name,
                        modifies_state=False,
                        is_controlled=False,
                        target_qubits=[tg],
                        control_qubits=None,
                        params=params,
                        shots=shots
                    )

                sg_method.__name__ = gate_name
                return sg_method

            setattr(cls, name, mk_sg_method(name))
        return cls

    return decorator

def tqc_nopar_gates(gate_names):
    """
    Make two-qubit controlled non-parametric gate methods.

    :param gate_names: strings with single gate names
    :return: decorator for target class
    """
    def decorator(cls):
        for name in gate_names:
            def mk_sg_method(gate_name):
                def sg_method(self, ct: int = 1, tg: int = 0, shots=None) -> Instruction:
                    return Instruction(
                        symbol=gate_name,
                        modifies_state=False,
                        is_controlled=True,
                        target_qubits=[tg],
                        control_qubits=[ct],
                        params=None,
                        shots=shots
                    )

                sg_method.__name__ = gate_name
                return sg_method

            setattr(cls, name, mk_sg_method(name))
        return cls

    return decorator

def tqc_par_gates(gate_names):
    """
    Make two-qubit parametric gate methods.

    :param gate_names: strings with single gate names
    :return: decorator for target class
    """
    def decorator(cls):
        for name in gate_names:
            def mk_sg_method(gate_name):
                def sg_method(self, ct: int = 1, tg: int = 0, params=None, shots=None) -> Instruction:
                    return Instruction(
                        symbol=gate_name,
                        modifies_state=False,
                        is_controlled=True,
                        target_qubits=[tg],
                        control_qubits=[ct],
                        params=params,
                        shots=shots
                    )

                sg_method.__name__ = gate_name
                return sg_method

            setattr(cls, name, mk_sg_method(name))
        return cls

    return decorator

def tests(gate_names):
    """
    Make single-qubit test methods.

    :param gate_names: strings with single gate names
    :return: decorator for target class
    """
    def decorator(cls):
        for name in gate_names:
            def mk_sg_method(gate_name):
                def sg_method(self, tgs: List[int] = None, params=None, shots=None) -> Instruction:
                    inst = Instruction(
                        symbol=gate_name,
                        modifies_state=False,
                        is_controlled=False,
                        target_qubits=tgs,
                        control_qubits=None,
                        params=params,
                        shots=shots
                    )

                    inst.instruction_type = InstructionType.TEST

                    return inst

                sg_method.__name__ = gate_name
                return sg_method

            setattr(cls, name, mk_sg_method(name))
        return cls

    return decorator

@sq_nopar_gates([ "x", "y", "z", "h", "s", "sdg", "t", "tdg" ])
@sq_par_gates(["p", "rx", "ry", "rz", "phase", "u2", "u3"])
@tqc_nopar_gates(["cx", "cy", "cz", "ch"])
@tqc_par_gates(["cp", "crx", "cry", "crz", "cphase", "cu"])
@tests(["resfreq", "satspect", "powrab", "pispec", "resspect", "dispshift", "rocalib"])
class ISA:
    """The Instruction Set Architecture comprises all possible operations that LCCF hardware
    will be able to make.
    """
    __circuit_instr = [
        "nop", "swap", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "p", "rx", "ry", "rz", "phase", "u2", "u3",
        "cx", "cy", "cz", "ch",
        "cp", "crx", "cry", "crz", "cphase", "cu",
        "measure", "reset"
    ]

    def __init__(self, name: str):
        self.name = name

    def swap(self, tg_a: int = 0, tg_b: int = 1, **kwargs) -> Instruction:
        """
        We define explicitly the swap gate due to its significance in the LCCF architecture and how
        it breaks the general pattern of other gates. Both qubits are targets since SWAP is symmetric.

        Accepts legacy ``ct``/``tg`` keyword arguments for backward compatibility.

        :param tg_a: first target qubit
        :param tg_b: second target qubit
        :return: SWAP instruction
        """
        # Backward compatibility: accept ct=/tg= keyword args
        if "ct" in kwargs:
            tg_a = kwargs["ct"]
        if "tg" in kwargs:
            tg_b = kwargs["tg"]

        return Instruction(
            symbol="swap",
            modifies_state=False,
            is_controlled=False,
            target_qubits=[tg_b],
            control_qubits=[tg_a],
            params=None,
            shots=None,
        )

    def nop(self, tgs=None) -> Instruction:
        """The nop instruction is quite peculiar in the sense that it is fungible, and can be used for
        various formal properties.

        :param tg: target qubits
        :return: an identity instruction
        """
        inst = Instruction(
            symbol="nop",
            modifies_state=False,
            is_controlled=False,
            target_qubits=tgs,
            control_qubits=None,
            params=None,
            shots=None,
        )

        # We want NOPs available and fungible with a general type
        inst.instruction_type = InstructionType.DELAYED
        return inst

    def measure(self, tgs: List[int]=None) -> Instruction:
        """Measure one or multiple qubits. Note that
        measurement modifies the state.

        :param tgs: qubits to measure
        :return: the measure instruction
        """
        inst = Instruction(
            symbol="measure",
            modifies_state=True,
            is_controlled=False,
            target_qubits=tgs,
            control_qubits=None,
            params=None,
            shots=None,
        )

        inst.instruction_type = InstructionType.CIRCUIT
        return inst

    def reset(self, tgs=None) -> Instruction:
        """Reset one or multiple qubits. Note that
        measurement modifies the state.

        Note: at the start,

        :param tgs: qubits to reset
        :return: the measure instruction
        """
        inst = Instruction(
            symbol="reset",
            modifies_state=True,
            is_controlled=False,
            target_qubits=tgs,
            control_qubits=None,
            params=None,
            shots=None,
        )

        # Resets may be used outside of circuits, delay until tested
        inst.instruction_type = InstructionType.DELAYED
        return inst

    def ftol(self, threshold_fidelity) -> Instruction:
        """Change the fidelity tolerance of the QPU as interpreted by the
        backend. The intent of this instruction is to determine when qubits
        define a functional QPU without raising an exception.

        :param threshold_fidelity:
        :return: instruction
        """
        inst = Instruction(
            symbol="ftol",
            modifies_state=True,
            is_controlled=False,
            target_qubits=None,
            control_qubits=None,
            params=[threshold_fidelity],
            shots=None,
        )

        # This is explicitly a control instruction, not to be used inside circuits
        inst.instruction_type = InstructionType.QPUSTATE
        return inst
