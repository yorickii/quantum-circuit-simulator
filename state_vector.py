from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from numpy.random import default_rng
from numpy.typing import NDArray
from typing import Self, TYPE_CHECKING

if TYPE_CHECKING:
    from operators import Operator

class StateVector():

    def __init__(self, num_qubits: int, state: NDArray[np.complex128]=None):
        """
        Initialize a state vector.

        Args:
            num_qubits: the number of qubits represented by the state vector.
            state: the desired state vector.
        """
        if num_qubits == 0:
            raise ValueError(f"State must contain at least one qubit.")
        
        # If no state is provided, initialize to the all up state
        if state is None:
            state = np.zeros(2**num_qubits, dtype=complex)
            state[0] = 1

        # If a state is provided, make sure it's the right length
        if len(state) != 2**num_qubits:
            raise ValueError(f"Must provide a state vector of length 2**{num_qubits}")
        
        # Make sure the provided state is normalized/normalizable
        magnitude = norm(state)
        if np.isclose(magnitude, 0):
            raise ValueError(f"State vector is non-normalizable. State: {state}. Norm: {magnitude}")
        
        state = state / magnitude

        self.num_qubits = num_qubits
        self.state = state

    
    @classmethod
    def from_index(cls, index: int, num_qubits: int) -> Self:
        """
        Initialize a quantum state basis vector from the provided index.
        """
        if index < 0:
            index = num_qubits + index

        if index < 0 or index >= 2**num_qubits:
            raise ValueError(f"Index out of range. Index: {index}. State vector length: {2**num_qubits}")

        state = np.zeros(2**num_qubits, dtype=complex)
        state[index] = 1

        return cls(num_qubits, state)
    
    @classmethod
    def from_basis_state(cls, bitstring: str) -> Self:
        """
        Initialize a quantum state basis vector from the bitstring describing the basis vector.
        """
        if not all(bit in "01" for bit in bitstring):
            raise ValueError("Basis state must be a string of 0s and 1s")
        
        num_qubits = len(bitstring)
        
        return cls.from_index(int(bitstring, 2), num_qubits)
    

    @classmethod
    def random_basis(cls, num_qubits: int, seed: int=None) -> Self:
        """
        Initialize a random basis state quantum state.
        """
        rng = default_rng(seed)

        index = rng.integers(0, 2**num_qubits)

        return cls.from_index(index, num_qubits)
        

    @classmethod
    def from_qubits(cls, initial_qubits: list[int | str | NDArray[np.complex128]]) -> Self:
        """
        Initialize a quantum state vector from a tensor product of provided one-qubit states.

        Args:
            initial_qubits: a list of the initial states of each qubit, if desired. Can be provided as
            "up"/"down" or 0/1 for each qubit, or a two-number numpy array for mixed states.
        
        Returns:
            a StateVector object generated from the provided qubits.
        """
        if len(initial_qubits) == 0:
            raise ValueError("Must provide at least one qubit")

        num_qubits = len(initial_qubits)

        qubits = []
        
        # Construct a list of numpy arrays to tensor product together. Will accept "up"/"down", 0/1, or
        # an explicit one-qubit state vector.
        for qubit in initial_qubits:
            if isinstance(qubit, str):
                match qubit.lower():
                    case "up":
                        qubits.append(np.array([1, 0],dtype=complex))
                    case "down":
                        qubits.append(np.array([0, 1],dtype=complex))
                    case _:
                        raise ValueError(f"Unrecognized state label: {qubit}")
                
            elif isinstance(qubit, int):
                match qubit:
                    case 0:
                        qubits.append(np.array([1, 0],dtype=complex))
                    case 1:
                        qubits.append(np.array([0, 1],dtype=complex))
                    case _:
                        raise ValueError(f"Unrecognized state label: {qubit}")
            
            elif isinstance(qubit, np.ndarray):
                if qubit.shape != (2,):
                    raise ValueError(f"Single-qubit state vectors must have shape (2,), got {qubit.shape}")
                
                magnitude = norm(qubit)
                if np.isclose(magnitude, 0):
                    raise ValueError(f"Cannot normalize a zero vector. Qubit: {qubit}. Norm: {magnitude}")
                qubits.append(qubit / magnitude)
            
            else:
                raise ValueError(f"Invalid state label type: {type(qubit)}")
            
        state = qubits[0]

        # Tensor product all the qubits together
        for qubit in qubits[1:]:
            state = np.kron(state, qubit)
        
        return cls(num_qubits, state)
        
    @classmethod
    def random_state(cls, num_qubits: int, seed: int=None) -> Self:
        """
        Initialize a completely random state vector of specified number of qubits.
        """
        rng = default_rng(seed)
        dim = 2**num_qubits

        # Generate random real and imaginary components
        real = rng.normal(size=dim)
        imag = rng.normal(size=dim)

        state = real + 1j * imag

        # Ensure normalization
        state = state / norm(state)

        return cls(num_qubits, state)
            
                    
    def evolve(self, operator: Operator, in_place: bool=True) -> NDArray:
        """
        Evolves the state vector by the provided operator. Operates in-place by default.

        Args:
            operator: the Operator object to evolve the state by.
            in_place: boolean flag controlling whether the evolution occurs in-place or not.
        
        Returns:
            new_state: the matrix product of the operator with the state vector.
        """
        # Ensure dimensions of operator and vector match
        expected = 2**self.num_qubits

        if operator.operator.shape != (expected, expected):
            raise ValueError(
                f"Operator shape {operator.operator.shape} incompatible with "
                f"{self.num_qubits} qubits."
            )

        # Operate on the state vector with the provided operator
        new_state = operator.operator @ self.state

        if in_place:
            self.state = new_state

        return new_state
    

    def measure_state(self, seed: int=None, in_place: bool=True) -> tuple[str, NDArray]:
        """
        Measure all qubits simultaneously.

        Args:
            seed: optional seed to control the randomness in the measuring process.
            in_place: boolean flag controlling whether the measurement occurs in-place or not.
        
        Returns:
            measurement: the bitstring (e.g. 101) of the basis state measured.
            new_state: the collapsed state vector after performing the measurement.
        """
        rng = default_rng(seed)

        probs = abs(self.state)**2
        probs /= sum(probs)
        measurement = rng.choice(len(self.state), p=probs)

        new_state = np.zeros_like(self.state, dtype=complex)
        new_state[measurement] = 1

        if in_place:
            self.state = new_state

        return format(measurement, f"0{self.num_qubits}b"), new_state
    
    
    def measure_qubit(self, qubit: int, seed: int=None, in_place: bool=True) -> tuple[int, NDArray]:
        """
        Measure the state of a specified qubit. Measures in-place by default.

        Args:
            qubit: the index of the qubit to beb measured. Zero-indexed from left; that is,
            in an n-qubit system the leftmost qubit is 0 and the rightmost is n-1.
            seed: optional seed to control the random choice of zero or one in the measuring process.
            in_place: boolean flag controlling whether the measurement occurs in-place or not.
        
        Returns:
            measurement: either 0 or 1, depending on the result of the measurement.
            new_state: the collapsed state vector after performing the desired measurement.
        """
        if qubit < 0:
            qubit = self.num_qubits + qubit
        
        if qubit >= self.num_qubits or qubit < 0:
            raise ValueError(f"Qubit index out of range. Number of qubits: {self.num_qubits}. Qubit index: {qubit}.")
        
        rng = default_rng(seed)

        p0 = 0
        p1 = 0

        indices = np.arange(len(self.state))

        # Do screwy bit operations on the indices to get an ordered list of zeros and ones for the specified qubit
        bits = (indices >> (self.num_qubits - 1 - qubit)) & 1
        # See README for a more detailed explanation of the reasoning behind this operation

        # Accumulate probabilities
        p0 = sum(abs(self.state[bits == 0])**2)
        p1 = sum(abs(self.state[bits == 1])**2)

        # Ensure probabilities sum to one
        probs = np.array([p0, p1])
        probs /= sum(probs)

        # Make the measurement
        measurement = rng.choice([0, 1], p=probs)

        # Zero out all components of the state where the qubit in question doesn't match the measurement
        mask = bits == measurement

        new_state = self.state.copy()
        new_state[~mask] = 0

        # Make sure the state is normalized
        new_state = new_state / norm(new_state)

        if in_place:
            self.state = new_state
        
        return measurement, new_state
    

    def copy(self) -> Self:
        """
        Create a deep copy of a state vector.
        """

        return type(self)(self.num_qubits, self.state.copy())