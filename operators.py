from __future__ import annotations
import numpy as np
from typing import Self, TYPE_CHECKING
from numpy.typing import NDArray
from state_vector import StateVector

H = 1/np.sqrt(2) * np.array([[1, 1],[1,-1]],dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
S = np.array([[1, 0],[0, 1j]], dtype=complex)
T = np.array([[1, 0],[0, np.exp(1j*np.pi/4)]], dtype=complex)

GATES = {
    "hadamard": H,
    "h": H,
    "x": X,
    "y": Y,
    "z": Z,

    "phase": S,
    "s": S,
    "t": T,

    "i": np.eye(2)
}

class Operator():
    
    def __init__(self, num_qubits: int, operator: NDArray[np.complex128]=None):
        """
        Initialize an operator.

        Args:
            num_qubits: the number of qubits the operator will be operating on.
            operator: the numpy array representing the operator.
        """

        # If no operator is provided, initialize to the identity
        if operator is None:
            operator = np.eye(2**num_qubits, dtype=complex)

        # Make sure the operator is the right dimension
        if np.shape(operator) != (2**num_qubits, 2**num_qubits):
            raise ValueError(f"Must provide an operator of shape (2**{num_qubits},2**{num_qubits})")
        
        # # Make sure the operator is unitary
        # maybe_eye = operator @ operator.conj().T
        # if not np.allclose(maybe_eye, np.eye(2**num_qubits, dtype=complex)):
        #     raise ValueError(f"Provided operator is not unitary. UU*: {maybe_eye}")
        
        self.num_qubits = num_qubits
        self.operator = operator

    
    @classmethod
    def identity(cls, num_qubits: int) -> Self:
        """
        Initialize an identity operator to act on the specified number of qubits.
        """

        return cls(num_qubits, np.eye(2**num_qubits, dtype=complex))
    
    @classmethod
    def hadamard(cls) -> Self:
        """
        Initialize a Hadamard gate.
        """

        return cls(1, GATES['hadamard'])
    

    @classmethod
    def pauli_x(cls) -> Self:
        """
        Initialize a Pauli X gate.
        """

        return cls(1, GATES['x'])
    

    @classmethod
    def pauli_y(cls) -> Self:
        """
        Initialize a Pauli Y gate.
        """

        return cls(1, GATES['y'])
    
    @classmethod
    def pauli_z(cls) -> Self:
        """
        Initialize a Pauli Z gate.
        """

        return cls(1, GATES['z'])
    

    @classmethod
    def phase(cls) -> Self:
        """
        Initialize a phase gate.
        """

        return cls(1, GATES['s'])
    
    @classmethod
    def t(cls) -> Self:
        """
        Initialize a T gate.
        """

        return cls(1, GATES['t'])
    
    
    @classmethod
    def rx(cls, theta: float) -> Self:
        """
        Initialize an x rotation gate.
        """

        return cls(1, np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                                [-1j * np.sin(theta / 2), np.cos(theta / 2)]]))
    

    @classmethod
    def ry(cls, theta: float) -> Self:
        """
        Initialize a y rotation gate.
        """

        return cls(1, np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                                [np.sin(theta / 2), np.cos(theta / 2)]]))
    


    @classmethod
    def rz(cls, theta: float) -> Self:
        """
        Initialize a z rotation gate.
        """

        return cls(1, np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]]))
    

    @classmethod
    def phase_shift(cls, phi: float) -> Self:
        """
        Initialize a phase shift gate.
        """

        return cls(1, np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex))
    

    @classmethod
    def u(cls, theta: float, phi: float, lam: float) -> Self:
        """
        Initialize an arbitrary U gate.
        """

        return cls(1, np.array([[np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]]))
    

    @classmethod
    def cnot(cls) -> Self:
        """
        Initialize a CNOT gate.
        """

        return cls(2, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    

    @classmethod
    def cz(cls) -> Self:
        """
        Initialize a CZ gate.
        """

        return cls(2, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]))
    

    @classmethod
    def swap(cls) -> Self:
        """
        Initialize a swap gate.
        """

        return cls(2, np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    
    
    @classmethod
    def toffoli(cls) -> Self:
        """
        Initialize a Toffoli gate.
        """

        return cls(3, np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]]))
    

    @classmethod
    def fredkin(cls) -> Self:
        """
        Initialize a Fredkin gate.
        """

        return cls(3, np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]]))
    

    @classmethod
    def from_factors(cls, factors: list[NDArray[np.complex128] | Self | str]) -> Self:
        """
        Initialize a multi-qubit operator from the provided list of single-qubit operators.

        Args:
            factors: a list of single qubit operators, provided as explicit numpy arrays or
            operator names (\"hadamard\", \"x\", \"y\", \"z\")

        Returns:
            The tensor product of the provided single-qubit operators. Product is conducted in order
            along the list.
        """
        if len(factors) == 0:
            raise ValueError("Must provide at least one factor")
        
        arrays = []

        for factor in factors:
            if isinstance(factor, str):
                try:
                    arrays.append(GATES[factor.lower()])
                except KeyError:
                    raise ValueError(f"Unrecognized operator name: {factor}")
            
            elif isinstance(factor, np.ndarray):
                if factor.shape != (2, 2):
                    raise ValueError(f"Expected a single-qubit operator of shape (2,2), got {factor.shape}")
                
                arrays.append(factor)
            
            elif isinstance(factor, Operator):
                if factor.num_qubits != 1:
                    raise ValueError(f"Expected a single-qubit operator, got a {factor.num_qubits} operator")
                
                arrays.append(factor.operator)

            else:
                raise ValueError(f"Invalid type {type(factor)} for an operator factor.")

        operator = arrays[0]

        for array in arrays[1:]:
            operator = np.kron(operator, array)

        return cls(len(factors), operator)
    

    @classmethod
    def _from_factors(cls, num_qubits: int, factors: list[NDArray[np.complex128] | Self]) -> Self:
        """
        Essentially the same as from_factors() but with fewer shape guardrails. Mainly meant for internal use.
        """
        if len(factors) == 0:
            raise ValueError("Must provide at least one factor")
        
        arrays = []

        for factor in factors:
            if isinstance(factor, np.ndarray):
                arrays.append(factor)
            
            elif isinstance(factor, Operator):
                arrays.append(factor.operator)

            else:
                raise ValueError(f"Invalid type {type(factor)} for an operator factor.")

        operator = arrays[0]

        for array in arrays[1:]:
            operator = np.kron(operator, array)

        return cls(num_qubits, operator)


    def tensor(self, other: Self) -> Self:

        op = np.kron(self.operator, other.operator)

        return Operator(self.num_qubits + other.num_qubits, op)
    

    def dagger(self) -> Self:
        """
        Compute the conjugate transpose of the operator.
        """
        return Operator(self.num_qubits, self.operator.conj().T)
    

    def __matmul__(self, other: Self | StateVector) -> Self:
        """
        Overload matrix multiplication to accept Operator objects.
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError(f"Shape mismatch: {self.num_qubits} != {other.num_qubits}")
        
        if isinstance(other, Operator):
            return Operator(self.num_qubits, self.operator @ other.operator)
        
        elif isinstance(other, StateVector):
            return StateVector(self.num_qubits, self.operator @ other.state)

        else:
            return NotImplemented


    def __mul__(self, other: int) -> Self:
        return Operator(self.num_qubits, other * self.operator)


    def __xor__(self, other: Self) -> Self:
        return self.tensor(other)


    def __str__(self):
        return str(self.operator)