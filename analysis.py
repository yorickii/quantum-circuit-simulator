from operators import Operator
from state_vector import StateVector
import numpy as np

def full_state_expectation(state: StateVector, operator: Operator) -> np.complex128:
        """
        Compute the expectation value of an operator.
        """
        if operator.num_qubits != state.num_qubits:
            raise ValueError("Operator size must match state vector size.",
                             f"Operator has {operator.num_qubits} qubits and state has {state.num_qubits} qubits.")

        return state @ (operator @ state)


def single_qubit_expectation(state: StateVector, operator: Operator, qubit: int) -> np.complex128:
    """
    Compute the expectation value of a single-qubit operator on part of the state.
    """
    factors = []
    for i in range(state.num_qubits):
        if i == qubit:
            factors.append(operator)
        else:
            factors.append("I")

    return full_state_expectation(state, Operator.from_factors(factors))