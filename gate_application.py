from dataclasses import dataclass
from operators import Operator

@dataclass
class GateApplication():
    """
    Tells the Circuit class how to use Operators.

    Attributes:
        operator: the Operator object to apply to the specified qubits.
        qubits: the qubit indices to apply the Operator to.
    """


    operator: Operator
    qubits: tuple[int,...]