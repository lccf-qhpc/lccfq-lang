"""
Filename: blocks.py
Author: Santiago Nunez-Corrales
Date: 2026-02-26
Version: 1.0
Description:
    This file provides algorithmic blocks that expand in place to allow for the construction
    of more complex programs at greater ease.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""

from enum import Enum
from typing import List

from ..arch.instruction import Instruction
from ..arch.register import QRegister, CRegister
from .preparation import prepare_basis, prepare_uniform, prepare_state


class BlockType(Enum):
    PREPAREBASIS = 1
    PREPAREUNIFORM = 2
    PREPARESTATE = 3
    ROTATE = 4
    ENTANGLESTEP = 5
    SWAP = 6
    ADD = 7
    MULTMOD = 8
    COMPARE = 9
    ORACLE = 10
    PHASEORACLE = 11
    QFT = 12
    IQFT = 13
    DIFFUSION = 14
    TIMEEVOLUTION = 15
    TROTTERSTEPS = 16
    HWEFFANSATZ = 17
    QAOASTEP = 18
    SYNDROME = 19
    TEMPLATE = 20


class BlockFactory:
    """
    The block factory creates procedural pieces used in a large variety of
    quantum algorithms.
    """

    def __init__(self, qreg: QRegister, creg: CRegister):
        self.qreg = qreg
        self.creg = creg

        self.__dispatch = {
            BlockType.PREPAREBASIS: self._prepare_basis,
            BlockType.PREPAREUNIFORM: self._prepare_uniform,
            BlockType.PREPARESTATE: self._prepare_state,
            BlockType.ROTATE: self._rotate,
            BlockType.ENTANGLESTEP: self._entangle_step,
            BlockType.SWAP: self._swap,
            BlockType.ADD: self._add,
            BlockType.MULTMOD: self._mult_mod,
            BlockType.COMPARE: self._compare,
            BlockType.ORACLE: self._oracle,
            BlockType.PHASEORACLE: self._phase_oracle,
            BlockType.QFT: self._qft,
            BlockType.IQFT: self._iqft,
            BlockType.DIFFUSION: self._diffusion,
            BlockType.TIMEEVOLUTION: self._time_evolution,
            BlockType.TROTTERSTEPS: self._trotter_steps,
            BlockType.HWEFFANSATZ: self._hw_eff_ansatz,
            BlockType.QAOASTEP: self._qaoa_step,
            BlockType.SYNDROME: self._syndrome,
            BlockType.TEMPLATE: self._template,
        }

    def block(self, b_type: BlockType, target, **kwargs) -> List[Instruction]:
        """Create a block of instructions for a given algorithmic primitive.

        :param b_type: block type intended for the register
        :param target: qubits involved in the block
        :param kwargs: additional arguments passed to the block
        :return: list of instructions composing the block
        :raises KeyError: if b_type is not a recognized BlockType
        """
        if b_type not in self.__dispatch:
            raise KeyError(f"Unknown block type '{b_type}'. Available: {list(self.__dispatch.keys())}")

        return self.__dispatch[b_type](target, **kwargs)

    def _prepare_basis(self, target, **kwargs) -> List[Instruction]:
        return prepare_basis(self.qreg.isa, target, **kwargs)

    def _prepare_uniform(self, target, **kwargs) -> List[Instruction]:
        return prepare_uniform(self.qreg.isa, target, **kwargs)

    def _prepare_state(self, target, **kwargs) -> List[Instruction]:
        return prepare_state(self.qreg.isa, target, **kwargs)

    def _rotate(self, target, **kwargs) -> List[Instruction]:
        pass

    def _entangle_step(self, target, **kwargs) -> List[Instruction]:
        pass

    def _swap(self, target, **kwargs) -> List[Instruction]:
        pass

    def _add(self, target, **kwargs) -> List[Instruction]:
        pass

    def _mult_mod(self, target, **kwargs) -> List[Instruction]:
        pass

    def _compare(self, target, **kwargs) -> List[Instruction]:
        pass

    def _oracle(self, target, **kwargs) -> List[Instruction]:
        pass

    def _phase_oracle(self, target, **kwargs) -> List[Instruction]:
        pass

    def _qft(self, target, **kwargs) -> List[Instruction]:
        pass

    def _iqft(self, target, **kwargs) -> List[Instruction]:
        pass

    def _diffusion(self, target, **kwargs) -> List[Instruction]:
        pass

    def _time_evolution(self, target, **kwargs) -> List[Instruction]:
        pass

    def _trotter_steps(self, target, **kwargs) -> List[Instruction]:
        pass

    def _hw_eff_ansatz(self, target, **kwargs) -> List[Instruction]:
        pass

    def _qaoa_step(self, target, **kwargs) -> List[Instruction]:
        pass

    def _syndrome(self, target, **kwargs) -> List[Instruction]:
        pass

    def _template(self, target, **kwargs) -> List[Instruction]:
        pass

