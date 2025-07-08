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
from .instruction import Instruction

### Generator for gate-based instructions

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
                        parameters=None,
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
                        parameters=params,
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
                        parameters=None,
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
                        parameters=params,
                        shots=shots
                    )

                sg_method.__name__ = gate_name
                return sg_method

            setattr(cls, name, mk_sg_method(name))
        return cls

    return decorator



@sq_nopar_gates([ "x", "y", "z", "h", "s", "sdg", "t", "tdg" ])
@sq_par_gates(["p", "rx", "ry", "rz", "phase", "u2", "u3"])
@tqc_nopar_gates(["cx", "cy", "cz", "ch"])
@tqc_par_gates(["cx", "cy", "cz", "ch"])
@tqc_par_gates(["cp", "crx", "cry", "crz", "cphase", "cu"])
class ISA:
    """The Instruction Set Architecture comprises all possible operations that LCCF hardware
    will be able to make.
    """

    def __init__(self, name: str):
        self.name = name

    def swap(self, tg_a: int, tg_b: int) -> Instruction:
        """
        We define explicitly the swap gate due to its significance in the LCCF architecture and how
        it breaks the general pattern of other gates.

        :param tg_a: target qubit a
        :param tg_b: target qubit b
        :return: SWAP instruction
        """
        return Instruction(
            symbol="swap",
            modifies_state=False,
            is_controlled=False,
            target_qubits=[tg_a, tg_b],
            control_qubits=None,
            parameters=None,
            shots=None,
        )

    def id(self, tgs=None) -> Instruction:
        """The identity instruction is quite peculiar in the sense that it is fungible, and can be used for
        various formal properties.

        :param tg: target qubits
        :return: an identity instruction
        """
        return Instruction(
            symbol="id",
            modifies_state=False,
            is_controlled=False,
            target_qubits=tgs,
            control_qubits=None,
            parameters=None,
            shots=None,
        )

    def measure(self, tgs=None) -> Instruction:
        """Measure one or multiple qubits. Note that
        measurement modifies the state.

        :param tgs: qubits to measure
        :return: the measure instruction
        """
        return Instruction(
            symbol="measure",
            modifies_state=True,
            is_controlled=False,
            target_qubits=tgs,
            control_qubits=None,
            parameters=None,
            shots=None,
        )

    def reset(self, tgs=None) -> Instruction:
        """Reset one or multiple qubits. Note that
        measurement modifies the state.

        Note: at the start,

        :param tgs: qubits to reset
        :return: the measure instruction
        """
        return Instruction(
            symbol="reset",
            modifies_state=True,
            is_controlled=False,
            target_qubits=tgs,
            control_qubits=None,
            parameters=None,
            shots=None,
        )
