"""
Filename: arch.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines the instruction set architecture of LCCFQ.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from lccfq_lang.arch.instruction import Instruction
from lccfq_lang.arch.error import BadParameterCount, UndefinedParametricInstruction


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
                        is_native=False,
                        modifies_state=False,
                        is_controlled=False,
                        target_qubits=[tg],
                        control_qubits=None,
                        parameters=None,
                        shots=shots
                    )

                sg_method.__name__ = gate_name
                return sg_method

            # Bind the method to the class, not as a static function
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
                        is_native=False,
                        modifies_state=False,
                        is_controlled=False,
                        target_qubits=[tg],
                        control_qubits=None,
                        parameters=params,
                        shots=shots
                    )

                sg_method.__name__ = gate_name
                return sg_method

            # Bind the method to the class, not as a static function
            setattr(cls, name, mk_sg_method(name))
        return cls

    return decorator


@sq_nopar_gates([ "x", "y", "z", "h", "s", "sdg", "t", "tdg" ])
@sq_par_gates(["p", "rx", "ry", "rz", "u2", "u3"])
class ISA:
    """The Instruction Set Architecture comprises all possible operations that LCCF hardware
    will be able to make.
    """

    def __init__(self, name: str):
        self.name = name

