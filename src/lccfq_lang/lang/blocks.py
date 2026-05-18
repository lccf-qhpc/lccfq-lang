"""
Filename: blocks.py
Author: Santiago Nunez-Corrales
Date: 2026-02-26
Version: 1.0
Description:
    BlockType enum and BlockFactory dispatcher. Per-family implementations live
    in sibling modules under lang/.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""

from enum import Enum
from typing import List

from ..arch.instruction import Instruction
from ..arch.register import QRegister, CRegister

from .preparation import prepare_basis, prepare_uniform, prepare_state
from .single_qubit import rotate
from .movement import swap, entangle_step
from .arithmetic import add, mult_mod, compare
from .oracles import oracle, phase_oracle
from .transforms import qft, iqft
from .amplification import diffusion
from .evolution import time_evolution, trotter_steps
from .variational import hw_eff_ansatz, qaoa_step
from .codes import syndrome
from .templates import template
from .multicontrol import mcx, mcz, mcry, mcrz


class BlockType(Enum):
    PREPARE_BASIS   = 1
    PREPARE_UNIFORM = 2
    PREPARE_STATE   = 3
    ROTATE          = 4
    ENTANGLE_STEP   = 5
    SWAP            = 6
    ADD             = 7
    MULT_MOD        = 8
    COMPARE         = 9
    ORACLE          = 10
    PHASE_ORACLE    = 11
    QFT             = 12
    IQFT            = 13
    DIFFUSION       = 14
    TIME_EVOLUTION  = 15
    TROTTER_STEPS   = 16
    HWEFF_ANSATZ    = 17
    QAOA_STEP       = 18
    SYNDROME        = 19
    TEMPLATE        = 20
    MCX             = 21
    MCZ             = 22
    MCRY            = 23
    MCRZ            = 24


class BlockFactory:
    """Dispatches BlockType requests to the per-family free function that builds the
    corresponding instruction list. All family functions share the signature
    (isa, target, **kwargs) -> List[Instruction]."""

    def __init__(self, qreg: QRegister, creg: CRegister):
        self.qreg = qreg
        self.creg = creg

        self.__dispatch = {
            BlockType.PREPARE_BASIS:   prepare_basis,
            BlockType.PREPARE_UNIFORM: prepare_uniform,
            BlockType.PREPARE_STATE:   prepare_state,
            BlockType.ROTATE:          rotate,
            BlockType.ENTANGLE_STEP:   entangle_step,
            BlockType.SWAP:            swap,
            BlockType.ADD:             add,
            BlockType.MULT_MOD:        mult_mod,
            BlockType.COMPARE:         compare,
            BlockType.ORACLE:          oracle,
            BlockType.PHASE_ORACLE:    phase_oracle,
            BlockType.QFT:             qft,
            BlockType.IQFT:            iqft,
            BlockType.DIFFUSION:       diffusion,
            BlockType.TIME_EVOLUTION:  time_evolution,
            BlockType.TROTTER_STEPS:   trotter_steps,
            BlockType.HWEFF_ANSATZ:    hw_eff_ansatz,
            BlockType.QAOA_STEP:       qaoa_step,
            BlockType.SYNDROME:        syndrome,
            BlockType.TEMPLATE:        template,
            BlockType.MCX:             mcx,
            BlockType.MCZ:             mcz,
            BlockType.MCRY:            mcry,
            BlockType.MCRZ:            mcrz,
        }

    def block(self, b_type: BlockType, target, **kwargs) -> List[Instruction]:
        """Build the instruction list for an algorithmic primitive.

        :raises KeyError: if b_type is not a recognized BlockType
        """
        if b_type not in self.__dispatch:
            raise KeyError(
                f"Unknown block type '{b_type}'. "
                f"Available: {list(self.__dispatch.keys())}"
            )
        return self.__dispatch[b_type](self.qreg.isa, target, **kwargs)
