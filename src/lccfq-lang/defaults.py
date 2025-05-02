"""
Filename: defaults.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides default values for system-level choices.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""

from dataclasses import dataclass


# Static data
from .mach.transpilers import XYiSW

transpilers = {
    "pfaff_v1": XYiSW
}


@dataclass
class Paths:
    """Default paths assumed by LCCFQ when installed as a library in user space.

    The default configuration will be determined by Pfaff Lab hardware specifications.
    """
    qpu_config: str = "./config/default.toml"


@dataclass
class Mach:
    transpiler: transpilers["pfaff_v1"]

