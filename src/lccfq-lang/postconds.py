"""
Filename: postconds.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines postconditions that instructions must satisfy
    after execution to be considered valid.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Postcondition:
    name: str
    description: str
    test: Callable
