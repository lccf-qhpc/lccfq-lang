"""
Filename: transpilers.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines the capability of a transpiler.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from typing import List
from abc import ABC, abstractmethod
from .ir import Gate
from ..arch.instruction import Instruction


class Transpiler(ABC):
    """
    A transpiler is an object that uses a mapping and a specific rendition of
    an instruction to produce native gates for a specific QPU.
    """

    def __init__(self):
        self.mapping = False
        self.topology = None
        self.table = None

    @abstractmethod
    def transpile_gate(self, instruction: Instruction) -> List[Gate]:
        pass

    @abstractmethod
    def transpile_test(self, instruction: Instruction) -> List[Gate]:
        pass
