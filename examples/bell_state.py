"""
Filename: bell_state.py
Author: Santiago Nunez-Corrales
Date: 2025-05-01
Version: 1.0
Description:
    This file provides an example that produces an entangled Bell state.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from lccfq_lang import QPU, QRegister, CRegister, Circuit, ISA


def bell_state():
    # Define the qpu and registers to use
    qpu = QPU(filename="../config/default.qpu")
    qreg = QRegister(2, qpu)
    creg = CRegister(size=2)
    isa = ISA("lccf")

    # Define a quantum circuit for bell states
    with Circuit(qreg, creg, shots=1000) as c:
        c >> isa.h(tg=0)
        c >> isa.cx(ct=0, tg=1)
        c >> isa.measure(tgs=[0, 1])

    print(creg.frequencies())


if __name__ == "__main__":
    bell_state()
