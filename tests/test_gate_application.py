from operators import Operator
from gate_application import GateApplication
import pytest
import re

def test_gate_application():
    # Test that gates cannot be applied to incorrect numbers of qubits
    h = Operator.hadamard()
    GateApplication(h, (1,)).qubits
    with pytest.raises(ValueError, match="Cannot apply a 1 qubit operator to 2 qubits."):
        GateApplication(h, (0,1))

    cnot = Operator.cnot()
    GateApplication(cnot, (0,1))
    gate = GateApplication(cnot, (2,0))
    with pytest.raises(ValueError, match="Cannot apply a 2 qubit operator to 1 qubits."):
        GateApplication(cnot, (0,))

    with pytest.raises(ValueError, match="Cannot apply a 2 qubit operator to 3 qubits."):
            GateApplication(cnot, (0,1,2))

    # Test that duplicate indices are rejected
    with pytest.raises(ValueError, match=re.escape("Duplicate qubit indices provided: (0, 0)")):
            GateApplication(cnot, (0,0))

    # Check that qubit ordering is preserved
    assert gate.qubits == (2, 0)