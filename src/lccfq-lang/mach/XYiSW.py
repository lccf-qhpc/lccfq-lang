"""
Filename: XYiSW.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides transpilation for devices using X, Y and sqrt(iSWAP) gates.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from typing import List
from .ir import Gate
from ..arch.instruction import Instruction


def transpile(instruction: Instruction) -> List[Gate]:
    """Transpile an instruction into a sequence of gates. The result is a list
    since gate ordering matters.

    :param instruction: The instruction to transpile.
    :return: A list of gates implementing that instruction.
    """
    pass
