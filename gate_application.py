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


    def __post_init__(self):
        if len(self.qubits) != self.operator.num_qubits:
            raise ValueError(f"Cannot apply a {self.operator.num_qubits} qubit operator to {len(self.qubits)} qubits.")

        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"Duplicate qubit indices provided: {self.qubits}")