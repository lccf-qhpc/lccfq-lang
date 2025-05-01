"""
Filename: preconds.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file defines preconditions that instructions must comply with
    to proceed successfully.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class Precondition:
    name: str
    description: str
    test: Callable
