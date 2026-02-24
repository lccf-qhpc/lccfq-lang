"""
Filename: grover.py
Author: Santiago Nunez-Corrales
Date: 2025-08-01
Version: 1.0
Description:
    This file provides an example using Grover's algorithm.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import numpy as np
from lccfq_lang import QPU, CRegister, Circuit


def grover():
    qpu = QPU(filename="config/default.toml")
    qreg = qpu.qregister(4)
    creg = CRegister(4)

    marked_bitstring = "1010"
    marked_bits = [int(b) for b in reversed(marked_bitstring)]
    n_qubits = 4
    n_iterations = int(round((np.pi / 4) * np.sqrt(2**n_qubits)))

    with Circuit(qreg, creg, qpu, shots=1000) as c:
        for q in range(n_qubits):
            c >> qpu.isa.h(tg=q)

        for _ in range(n_iterations):
            for i, bit in enumerate(marked_bits):
                if bit == 0:
                    c >> qpu.isa.x(tg=i)

            c >> qpu.isa.h(tg=3)
            c >> qpu.isa.cx(ct=2, tg=3)
            c >> qpu.isa.cx(ct=1, tg=2)
            c >> qpu.isa.cx(ct=0, tg=1)
            c >> qpu.isa.x(tg=0)
            c >> qpu.isa.cx(ct=1, tg=0)
            c >> qpu.isa.x(tg=0)
            c >> qpu.isa.cx(ct=0, tg=1)
            c >> qpu.isa.cx(ct=1, tg=2)
            c >> qpu.isa.cx(ct=2, tg=3)
            c >> qpu.isa.h(tg=3)

            for i, bit in enumerate(marked_bits):
                if bit == 0:
                    c >> qpu.isa.x(tg=i)

            for q in range(n_qubits):
                c >> qpu.isa.h(tg=q)
                c >> qpu.isa.x(tg=q)

            c >> qpu.isa.h(tg=3)
            c >> qpu.isa.cx(ct=2, tg=3)
            c >> qpu.isa.cx(ct=1, tg=2)
            c >> qpu.isa.cx(ct=0, tg=1)
            c >> qpu.isa.x(tg=0)
            c >> qpu.isa.cx(ct=1, tg=0)
            c >> qpu.isa.x(tg=0)
            c >> qpu.isa.cx(ct=0, tg=1)
            c >> qpu.isa.cx(ct=1, tg=2)
            c >> qpu.isa.cx(ct=2, tg=3)
            c >> qpu.isa.h(tg=3)

            for q in range(n_qubits):
                c >> qpu.isa.x(tg=q)
                c >> qpu.isa.h(tg=q)

        c >> qpu.isa.measure(tgs=list(range(n_qubits)))

    print(creg.frequencies())

if __name__ == "__main__":
    grover()
