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
from lccfq_lang import QPU, CRegister, Circuit


def bell_state():
    # Define the qpu and registers to use
    qpu = QPU(filename="config/default.toml")
    qreg = qpu.qregister(2)
    creg = CRegister(size=2)

    # Define a quantum circuit for bell states
    with Circuit(qreg, creg, qpu, shots=1000) as c:
        c >> qpu.isa.h(tg=0)
        c >> qpu.isa.cx(ct=0, tg=1)
        c >> qpu.isa.measure(tgs=[0, 1])

    print(creg.frequencies())


if __name__ == "__main__":
    bell_state()
