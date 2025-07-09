"""
Filename: ir.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides entities involved in the LCCFQ intermediate representation

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from abc import ABC
from typing import List


class Command(ABC):
    """Abstract definition of a command.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol


class Gate(Command):
    """
    Gates assume that the ordering of application in the circuit is the same as the
    diagram of that circuit.
    """
    def __init__(self,
                 symbol: str,
                 target_qubits: List[int],
                 control_qubits: List[int],
                 params: List[float]):
        super().__init__(symbol)
        self.target_qubits = target_qubits
        self.control_qubits = control_qubits
        self.params = params

    def __repr__(self):
        return f"G: {self.symbol} @ {self.target_qubits} ctrl by {self.control_qubits} w/ params={self.params}"

    def to_json(self):
        """Provide a serializable JSON representation of a gate to cross the backend.

        :return: a JSON stub
        """
        return {
            "symbol": self.symbol,
            "target_qubits": self.target_qubits,
            "control_qubits": self.control_qubits,
            "params": self.params
        }


class Control(Command):
    """Define a control instruction. Control instructions contain parameters that
    modulate their behavior.

    """
    def __init__(self,
                 symbol: str,
                 params: List[int]):
        super().__init__(symbol)
        self.params = params

    def to_json(self):
        """Provide a serializable JSON representation of a control command to cross the backend.

        :return: a JSON stub
        """
        return {
            "symbol": self.symbol,
            "params": self.params
        }



class Test(Command):
    """Define a test instruction. Tests can be parametric and require specifying the number of
    shots required to obtain meaningful statistics.
    """
    def __init__(self,
                 symbol: str,
                 params: List[int],
                 shots: int):
        super().__init__(symbol)
        self.params = params
        self.shots = shots

    def to_json(self):
        """Provide a serializable JSON representation of a test command to cross the backend.

        :return: a JSON stub
        """

        return {
            "symbol": self.symbol,
            "params": self.params,
            "shots": self.shots
        }
