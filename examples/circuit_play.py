import logging
import math

from src.lccfq_lang import QPU, CRegister, ISA, Circuit

logging.basicConfig(level=logging.DEBUG)


def x_gate():
    qpu = QPU(filename="config/default.toml", last_pass="executed")
    qreg = qpu.qregister(4)
    creg = CRegister(1)
    isa = ISA("lccf")

    with Circuit(qreg, creg, qpu=qpu, shots=1000) as c:
        c >> isa.x(tg=0)
        c >> isa.measure(tgs=[0])

    return c


def two_x_gates():
    qpu = QPU(filename="config/default.toml", last_pass="executed")
    qreg = qpu.qregister(4)
    creg = CRegister(1)
    isa = ISA("lccf")

    with Circuit(qreg, creg, qpu=qpu, shots=1000) as c:
        c >> isa.x(tg=0)
        c >> isa.x(tg=0)
        c >> isa.measure(tgs=[0])

    return c


def two_x_gates_333():
    qpu = QPU(filename="config/default.toml", last_pass="executed")
    qreg = qpu.qregister(4)
    creg = CRegister(1)
    isa = ISA("lccf")

    with Circuit(qreg, creg, qpu=qpu, shots=333) as c:
        c >> isa.x(tg=0)
        c >> isa.x(tg=0)
        c >> isa.measure(tgs=[0])

    return c


def rx_half_pi():
    qpu = QPU(filename="config/default.toml", last_pass="executed")
    qreg = qpu.qregister(4)
    creg = CRegister(1)
    isa = ISA("lccf")

    with Circuit(qreg, creg, qpu=qpu, shots=1000) as c:
        c >> isa.rx(tg=0, params=[math.pi / 2])
        c >> isa.measure(tgs=[0])

    return c


if __name__ == "__main__":
    print("=== X gate ===")
    c = x_gate()
    print("Raw bitstring counts:", c.results())
    print("Frequencies:", c.frequencies())

    print("\n=== Two X gates ===")
    c = two_x_gates()
    print("Raw bitstring counts:", c.results())
    print("Frequencies:", c.frequencies())

    print("\n=== Two X gates ===")
    c = two_x_gates_333()
    print("Raw bitstring counts:", c.results())
    print("Frequencies:", c.frequencies())

    print("\n=== Rx(pi/2) ===")
    c = rx_half_pi()
    print("Raw bitstring counts:", c.results())
    print("Frequencies:", c.frequencies())
