from operators import Operator
from gate_application import GateApplication
from state_vector import StateVector
from numpy import eye, array, ndarray, arange
from numpy.typing import NDArray

class Circuit():

    def __init__(self, gates: list[GateApplication], num_qubits: int):
        
        self.gates = gates
        self.num_qubits = num_qubits
        self.operators: list[Operator | list[int]] = []

    
    def simulate(self, initial_state: StateVector) -> StateVector:
        """
        Simulate the action of the compiled circuit on a StateVector.
        """
        state = initial_state.copy()

        if not self.operators:
            raise ValueError("Circuit must be compiled before simulating.")
        
        for operator in self.operators:
            # Operators in the circuit are either Operator objects or permutation vectors. Pick the right action.
            if isinstance(operator, list | ndarray) and array(operator).ndim == 1:
                state.permute_state(operator)

            elif isinstance(operator, Operator):
                state.evolve(operator)

            else:
                raise ValueError(f"Invalid operator in circuit: {operator}")

        return state

    
    def compile(self):
        """
        Turn the circuit's list of GateApplications into a list of operators that can actually be applied to
        a StateVector.
        """
        for gate in self.gates:
            qubits = gate.qubits
            operator = gate.operator
            operator_size = operator.num_qubits

            # If the operator in a gate is built for more than one qubit, we need to be careful about how
            # we handle it--all the qubits it will act on need to be moved to be adjacent to each other.
            if operator_size > 1:
                # Construct a permutation vector for the qubits themselves. This does not permute the state vector;
                # just the vector that calls things "qubit 0", "qubit 1", etc.
                permuted_qubits = list(qubits) + [i for i in sorted((set(range(self.num_qubits)) - set(qubits)))]

                # Construct a permutation vector for the state vector. See README.md for extended explanation of
                # the bit manipulations happening here.
                indices = arange(2**self.num_qubits)
                bits = (indices[:, None] >> (self.num_qubits - 1 - array(permuted_qubits))) & 1
                permuted_state_indices = (bits << (self.num_qubits - 1 - arange(self.num_qubits))).sum(axis=1)

                # Swap the qubits around, use the gate, then swap the qubits back.
                self.operators.append(permuted_state_indices)
                self.operators.append(self.construct_operator(gate, permuted_qubits))
                self.operators.append(self.invert_permutation(permuted_state_indices))

            # If the operator only acts on one qubit at a time, just add it to the list.
            else:
                self.operators.append(self.construct_operator(gate, range(self.num_qubits)))

    
    def construct_operator(self, gate: GateApplication, ordered_qubits: list[int]) -> Operator:
        """
        Constructs an n-qubit operator from a GateApplication, assuming that any multi-qubit gates are only applied
        to adjacent qubits.
        """
        qubits = gate.qubits
        operator = gate.operator
        operator_size = operator.num_qubits

        operator_queued = False

        factors = []

        # Go through the qubit indices. When we hit one the passed operator is supposed to act on, add the operator
        # to the list. Otherwise, add the identity.
        for i in ordered_qubits:
            if i in qubits:
                # If we're constructing an operator for a multi-qubit gate (like CNOT or something), only add the gate
                # to the list once.
                if not operator_queued:
                    factors.append(operator.operator)
                    if operator_size > 1:
                        operator_queued = True
            else:
                factors.append(eye(2))

        return Operator._from_factors(self.num_qubits,factors)


    def invert_permutation(self, permutation_vector: list[int]) -> list[int]:
        """
        Compute a permutation vector to invert a given permutation.
        """
        inverse = [0] * len(permutation_vector)

        for i, j in enumerate(permutation_vector):
            inverse[j] = i

        return inverse