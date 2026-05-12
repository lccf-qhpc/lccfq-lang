"""
Filename: __init__.py
Author: Santiago Nunez-Corrales
Date: 2026-02-26
Version: 1.0
Description:
    Public surface for the lccfq_lang.lang package. Exposes only BlockType and
    BlockFactory; family modules (preparation, single_qubit, movement, ...)
    are accessed by importing them directly.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""

from .blocks import BlockType, BlockFactory

__all__ = ["BlockType", "BlockFactory"]
