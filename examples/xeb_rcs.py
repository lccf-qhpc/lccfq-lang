"""
Filename: xeb_rcs.py
Author: Santiago Nunez-Corrales
Date: 2025-08-01
Version: 1.0
Description:
    This file provides an example implementation of cross-entropy benchmarking through
    random circuit sampling.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import random
import numpy as np

from collections import Counter
from .xeb_util import XEBSimulator
from lccfq_lang import QPU, CRegister, Circuit, ISA


def random_sqg(isa, qidx):
    """Yield a random single-qubit instruction

    :param isa: instruction set architecture instance
    :param qidx: target qubit index
    :return: random instruction
    """
    gates = [isa.x, isa.y, isa.z, isa.h, isa.s, isa.t]
    return random.choice(gates)(tg=qidx)


def random_tqg(isa, qidx0, qidx1):
    """Yield a random CX instruction with two qubits.

    :param isa: instruction set architecture instance
    :param qidx0: qubit index 0
    :param qidx1: qubit index 1

    :return: CX instruction
    """
    if random.choice([True, False]):
        return isa.cx(ct=qidx0, tg=qidx1)
    else:
        return isa.cx(ct=qidx1, tg=qidx0)


def generate_rand_inst(n_qubits, depth, isa):
    """ Generate a random instructions using single- and two-qubit instructions.

    :param n_qubits: number of qubits
    :param depth: circuit depth in 1:2-qubit layers
    :param isa: instruction set architecture instance
    :return: circuit
    """
    circuit = []

    for layer in range(depth):
        # single-qubit layer
        for q in range(n_qubits):
            circuit.append(random_sqg(isa, q))

        # two-qubit layer
        pairs = list(zip(range(0, n_qubits - 1, 2), range(1, n_qubits, 2)))
        random.shuffle(pairs)

        for q0, q1 in pairs:
            circuit.append(random_tqg(isa, q0, q1))

    return circuit


def compute_xeb(result: dict, ideal_probs: dict, n_qubits: int) -> float:
    """Compute the XEB fidelity.

    :param result: result computed through RCS
    :param ideal_probs: expected probability distribution
    :param n_qubits: number of qubits
    :return: fidelity
    """
    total = sum(result.values())
    sum_probs = 0.0

    for bitstring, count in result.items():
        p = ideal_probs.get(bitstring, 0)
        sum_probs += p * count

    average_p = sum_probs / total
    fidelity = (2**n_qubits) * average_p - 1
    return fidelity


def xeb_rcs(n_qubits=5, depth=20, shots=1000):
    """Obtain the XEB fidelity using random circuit sampling.

    :param n_qubits: number of qubits
    :param depth: number of 1:2-qubit layers
    :param shots: number of shots
    :return:
    """
    qpu = QPU(filename="../config/default.qpu")
    qreg = qpu.qregister(n_qubits)
    creg = CRegister(n_qubits)
    isa = ISA("lccf")

    rcs_circuit = generate_rand_inst(n_qubits, depth, isa)

    with Circuit(qreg, creg, qpu, shots=shots) as c:
        for gate in rcs_circuit:
            c >> gate
        c >> isa.measure(tgs=list(range(n_qubits)))

    result = creg.frequencies()

    # Use an internal state vector to simulate the resulting probabilities
    ideal_probs = XEBSimulator().probabilities(rcs_circuit, n_qubits)

    # Contrast against
    fidelity = compute_xeb(result, ideal_probs, n_qubits)

    print("Ideal circuit frequencies:", result)
    print("XEB fidelity:", fidelity)