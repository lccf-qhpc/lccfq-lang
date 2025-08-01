"""
Filename: dj.py
Author: Santiago Nunez-Corrales
Date: 2025-08-01
Version: 1.0
Description:
    This file provides an example implementing the Deutsch-Jozsa algorithm.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
from lccfq_lang import QPU, QRegister, CRegister, Circuit, ISA


def deutsch_jozsa(n: int = 3, oracle_type: str = "balanced"):
    """Implementation of the Deutsch-Jozsa algorithm.

    :param n: number of input qubits
    :param oracle_type: constant or balanced
    :return: nothing
    """
    qpu = QPU(filename="../config/default.qpu")
    qreg = QRegister(n + 1, qpu)
    creg = CRegister(n)
    isa = ISA("lccf")

    with Circuit(qreg, creg, shots=1000) as c:
        for i in range(n):
            c >> isa.h(tg=i)

        c >> isa.x(tg=n)
        c >> isa.h(tg=n)

        if oracle_type == "constant":
            # Nothing to be done for identity function
            pass

        elif oracle_type == "balanced":
            # Standard XOR oracle
            for i in range(n):
                c >> isa.cx(ct=i, tg=n)
        else:
            raise ValueError("oracle_type must be 'constant' or 'balanced'")

        for i in range(n):
            c >> isa.h(tg=i)

        c >> isa.measure(tgs=list(range(n)))

    result = creg.frequencies()

    if all(outcome == "0" * n for outcome in result):
        print("Oracle function is constant.")
    else:
        print("Oracle function is balanced.")


if __name__ == "__main__":
    deutsch_jozsa(n=3, oracle_type="balanced")