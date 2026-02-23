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
from lccfq_lang import QPU, CRegister, Circuit, ISA


def grover():
    qpu = QPU(filename="../config/default.qpu")
    qreg = qpu.qregister(4)
    creg = CRegister(4)
    isa = ISA("lccf")

    marked_bitstring = "1010"
    marked_bits = [int(b) for b in reversed(marked_bitstring)]
    n_qubits = 4
    n_iterations = int(round((np.pi / 4) * np.sqrt(2**n_qubits)))

    with Circuit(qreg, creg, qpu, shots=1000) as c:
        for q in range(n_qubits):
            c >> isa.h(tg=q)

        for _ in range(n_iterations):
            for i, bit in enumerate(marked_bits):
                if bit == 0:
                    c >> isa.x(tg=i)

            c >> isa.h(tg=3)
            c >> isa.cx(ct=2, tg=3)
            c >> isa.cx(ct=1, tg=2)
            c >> isa.cx(ct=0, tg=1)
            c >> isa.x(tg=0)
            c >> isa.cx(ct=1, tg=0)
            c >> isa.x(tg=0)
            c >> isa.cx(ct=0, tg=1)
            c >> isa.cx(ct=1, tg=2)
            c >> isa.cx(ct=2, tg=3)
            c >> isa.h(tg=3)

            for i, bit in enumerate(marked_bits):
                if bit == 0:
                    c >> isa.x(tg=i)

            for q in range(n_qubits):
                c >> isa.h(tg=q)
                c >> isa.x(tg=q)

            c >> isa.h(tg=3)
            c >> isa.cx(ct=2, tg=3)
            c >> isa.cx(ct=1, tg=2)
            c >> isa.cx(ct=0, tg=1)
            c >> isa.x(tg=0)
            c >> isa.cx(ct=1, tg=0)
            c >> isa.x(tg=0)
            c >> isa.cx(ct=0, tg=1)
            c >> isa.cx(ct=1, tg=2)
            c >> isa.cx(ct=2, tg=3)
            c >> isa.h(tg=3)

            for q in range(n_qubits):
                c >> isa.x(tg=q)
                c >> isa.h(tg=q)

        c >> isa.measure(tgs=list(range(n_qubits)))

    print(creg.frequencies())

if __name__ == "__main__":
    grover()
