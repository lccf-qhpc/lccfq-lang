"""
Filename: teleportation.py
Author: Santiago Nunez-Corrales
Date: 2025-08-01
Version: 1.0
Description:
    This file provides an example implementing quantum teleportation.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from lccfq_lang import QPU, QRegister, CRegister, Circuit, ISA, QASMSynthesizer


def quantum_teleportation():
    """
    Quantum teleportation protocol implementation.

    :return: nothing
    """
    qpu = QPU(filename="config/default.toml", last_pass="transpiled")
    qreg = QRegister(3, qpu)
    creg = CRegister(2)
    isa = ISA("lccf")

    # Setup:
    #
    # q0 = Alice's qubit
    # q1 = shared entanglement qubit
    # q2 = Bob's qubit

    with Circuit(qreg, creg, shots=1000, verbose=True) as c:
        c >> isa.h(tg=0)
        c >> isa.h(tg=1)
        c >> isa.cx(ct=1, tg=2)
        c >> isa.cx(ct=0, tg=1)
        c >> isa.h(tg=0)

        # Note that in this case, we have no direct way of applying classical
        # logic into quantum code to simplify LCCFQ's design. Results must be
        # obtained through post-processing.
        c >> isa.measure(tgs=[2, 1, 0])

    # Synthesize OpenQASM code from the circuit
    synth = QASMSynthesizer()
    synth.synth_circuit(circuit=c, path="./teleport.qasm")

    freqs = creg.frequencies()
    corrected = postprocess(freqs)
    print(corrected)


def postprocess(freqs: dict) -> dict:
    """Post-processing code to apply corrections to Bob's qubit.

    :param freqs: frequencies returned after successful quantum teleportation protocol
    :return: corrected frequencies after postprocessing
    """
    corrected = {"0": 0, "1": 0}

    for outcome, count in freqs.items():
        if len(outcome) != 3:
            continue

        q2_bit = int(outcome[0])
        c1 = int(outcome[1])
        c0 = int(outcome[2])
        result = q2_bit

        # c1 == 1 -> X(outcome)
        if c1 == 1:
            result ^= 1

        # c0 == 1 -> Z(outcome)
        if c0 == 1:
            result ^= 1

        corrected[str(result)] += count

    return corrected


if __name__ == "__main__":
    quantum_teleportation()