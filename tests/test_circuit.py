from state_vector import StateVector
from operators import Operator
from circuit import Circuit
import numpy as np
from numpy.random import default_rng, Generator
import pytest

def test_compile():
    # Test that a single qubit circuit compiles correctly
    qc = Circuit(1).h(0)
    qc.compile()
    assert len(qc.instructions) == 1
    assert qc.instructions[0] == Operator.hadamard()

    # Test that a multi qubit circuit with multiple single qubit gates compiles correctly
    qc = Circuit(3).h(0).x(1).rx(2, 0.25)
    gates = qc.gates.copy()
    qc.compile()
    assert len(qc.instructions) == 3
    assert qc.instructions[0] == Operator.hadamard() ^ Operator(1) ^ Operator(1)
    assert qc.instructions[1] == Operator(1) ^ Operator.pauli_x() ^ Operator(1)
    assert qc.instructions[2] == Operator(1) ^ Operator(1) ^ Operator.rx(0.25)

    # Test that compilation does not change the original list of gates (assuming no optimizations)
    assert qc.gates == gates

    # Test that changing the circuit changes the compiled list of instructions
    instructions = qc.instructions.copy()
    qc.h(1)
    qc.compile()
    assert qc.instructions != instructions

    # More multi qubit circuit single qubit gates testing
    qc = Circuit(3).h(0).x(0).x(1).y(2)
    qc.compile()
    assert len(qc.instructions) == 4
    assert qc.instructions[0] == Operator.hadamard() ^ Operator(1) ^ Operator(1)
    assert qc.instructions[1] == Operator.pauli_x() ^ Operator(1) ^ Operator(1)
    assert qc.instructions[2] == Operator(1) ^ Operator.pauli_x() ^ Operator(1)
    assert qc.instructions[3] == Operator(1) ^ Operator(1) ^ Operator.pauli_y()

    # Test that multi qubit gates compile correctly
    qc = Circuit(3).cnot(0,1)
    qc.compile()
    assert len(qc.instructions) == 1
    assert qc.instructions[0] == Operator.cnot() ^ Operator(1)

    qc = Circuit(3).cnot(2,0)
    qc.compile()
    assert len(qc.instructions) == 3
    assert (qc.instructions[0] == np.array([0, 4, 1, 5, 2, 6, 3, 7])).all()
    assert qc.instructions[1] == Operator.cnot() ^ Operator(1)
    assert (qc.instructions[2] == np.array([0, 2, 4, 6, 1, 3, 5, 7])).all()

    qc = Circuit(3).toffoli(0,1,2)
    qc.compile()
    assert len(qc.instructions) == 1
    assert qc.instructions[0] == Operator.toffoli()

    qc = Circuit(3).toffoli(2,0,1)
    qc.compile()
    assert (qc.instructions[0] == np.array([0, 4, 1, 5, 2, 6, 3, 7])).all()
    assert qc.instructions[1] == Operator.toffoli()
    assert (qc.instructions[2] == np.array([0, 2, 4, 6, 1, 3, 5, 7])).all()

    # Test rejection of invalid indices
    with pytest.raises(ValueError, match="Qubit index out of range. Maximum index: 1"):
        Circuit(2).h(2).compile()

def test_simulate():
    qc = Circuit(2).h(0).cnot(0,1).x(1)
    vec = StateVector.from_basis_state('00')

    # Test rejection of uncompiled circuit
    with pytest.raises(ValueError, match="Circuit must be compiled before simulating."):
        qc.simulate(vec)

    qc.compile()

    # Test rejection of misshapen state vectors
    bad_vec = StateVector.from_basis_state('0')

    with pytest.raises(ValueError, match="Circuit expects a 2 qubit state vector, got 1"):
        qc.simulate(bad_vec)

    # Test that a particular circuit produces the expected result
    qc = Circuit(3).h(0).x(1).cnot(0,2).h(2)
    qc.compile()
    vec = StateVector.from_basis_state('000')

    result = qc.simulate(vec)

    assert np.allclose(result.state, 1 / 2 * np.array([0,0,1,1,0,0,1,-1]))

    # Test that the original state is not modified
    assert vec != result
    assert (vec.state == np.array([1,0,0,0,0,0,0,0])).all()

    # Test that a circuit and equivalent full state operator produce identical results
    qc = Circuit(3).h(0).x(2).cnot(0,1).h(1)
    qc.compile()

    op = ((Operator(1) ^ Operator.hadamard() ^ Operator(1)) @ (Operator.cnot() ^ Operator(1)) 
      @ (Operator(1) ^ Operator(1) ^ Operator.pauli_x()) @ (Operator.hadamard() ^ Operator(1) ^ Operator(1)))

    assert np.allclose(qc.simulate(vec).state, (op @ vec).state)

def test_single_qubit_optimize():
    # Test single qubit involutary gates
    qc = Circuit(1).h(0).h(0)
    qc.compile()
    assert len(qc.instructions) == 0

    qc = Circuit(1).x(0).x(0)
    qc.compile()
    assert len(qc.instructions) == 0

    qc = Circuit(1).y(0).y(0)
    qc.compile()
    assert len(qc.instructions) == 0

    qc = Circuit(1).z(0).z(0)
    qc.compile()
    assert len(qc.instructions) == 0

    # Test that non-identical gates do not cancel
    qc = Circuit(1).h(0).x(0)
    qc.compile()
    assert len(qc.instructions) == 2

    # Test that three involutary gates reduce to one
    qc = Circuit(1).h(0).h(0).h(0)
    qc.compile()
    assert len(qc.instructions) == 1

    # Test that four involutary gates reduce to none
    qc = Circuit(1).h(0).h(0).h(0).h(0)
    qc.compile()
    assert len(qc.instructions) == 0

    # Test that identities don't break cancelations
    qc = Circuit(1).h(0).identity(0).h(0)
    qc.compile()
    assert len(qc.instructions) == 0

    # Test rotation merges
    qc = Circuit(1).rx(0, 0.25).rx(0, 0.25)
    qc.compile()
    assert len(qc.instructions) == 1

    qc = Circuit(1).ry(0, 0.25).ry(0, 0.25)
    qc.compile()
    assert len(qc.instructions) == 1

    qc = Circuit(1).rz(0, 0.25).rz(0, 0.25)
    qc.compile()
    assert len(qc.instructions) == 1

    qc = Circuit(1).rx(0, 0.25).rx(0, 0.25).rx(0, 0.25)
    qc.compile()
    assert len(qc.instructions) == 1

    # Test that multiple merges is equivalent to just multiplying by a big operator
    vec = StateVector.from_basis_state('0')
    op = Operator.rx(0.25) @ Operator.rx(0.25) @ Operator.rx(0.25)
    assert np.allclose(qc.simulate(vec).state, (op @ vec).state)

    # Make sure rotation symbols are updated correctly
    assert qc.gates[0].operator.symbol == 'Rx(\u03B8)'

    # Test that identities don't break merges
    qc = Circuit(1).rx(0, 0.25).identity(0).rx(0, 0.25)
    qc.compile()
    assert len(qc.instructions) == 1

    # Test that rotations only merge along identical axes
    qc = Circuit(1).rx(0, 0.25).ry(0, 0.25)
    qc.compile()
    assert len(qc.instructions) == 2

    # Test that rotations don't merge across other gates
    qc = Circuit(1).rx(0, 0.25).h(0).rx(0, 0.25)
    qc.compile()
    assert len(qc.instructions) == 3

def test_multi_qubit_optimize():
    # Test multi qubit involutary gate cancelations
    qc = Circuit(2).cnot(0,1).cnot(0,1)
    qc.compile()
    assert len(qc.instructions) == 0

    qc = Circuit(2).cz(0,1).cz(0,1)
    qc.compile()
    assert len(qc.instructions) == 0

    qc = Circuit(2).swap(0,1).swap(0,1)
    qc.compile()
    assert len(qc.instructions) == 0

    qc = Circuit(3).toffoli(0,1,2).toffoli(0,1,2)
    qc.compile()
    assert len(qc.instructions) == 0

    qc = Circuit(3).fredkin(0,1,2).fredkin(0,1,2)
    qc.compile()
    assert len(qc.instructions) == 0

    # Test that cancelation requires identical qubit ordering
    qc = Circuit(3).toffoli(0,1,2).toffoli(1,0,2)
    qc.compile()
    assert len(qc.instructions) == 4    # one toffoli, a permutation, a toffoli, and an inverse permutation

    # Test that cancelation does not occur across other gates
    qc = Circuit(3).toffoli(0,1,2).h(1).toffoli(0,1,2)
    qc.compile()
    assert len(qc.instructions) == 3

    # Test that cancelations can occur as long as intervening operations are irrelevant
    qc = Circuit(3).cnot(0,2).h(1).cnot(0,2)
    qc.compile()
    assert len(qc.instructions) == 1

    # Three involutary gates reduce to one
    qc = Circuit(3).toffoli(0,1,2).toffoli(0,1,2).toffoli(0,1,2)
    qc.compile()
    assert len(qc.instructions) == 1

    # Four involutary gates reduce to none
    qc = Circuit(3).toffoli(0,1,2).toffoli(0,1,2).toffoli(0,1,2).toffoli(0,1,2)
    qc.compile()
    assert len(qc.instructions) == 0

def test_end_to_end_correctness():
    # Test Bell state construction
    qc = Circuit(2).h(0).cnot(0,1)
    vec = StateVector.from_basis_state('00')
    expected = 1 / np.sqrt(2) * np.array([1,0,0,1])
    qc.compile()

    assert np.allclose(qc.simulate(vec).state, expected)

    # Reverse the qubits, should still produce the same Bell state
    qc = Circuit(2).h(1).cnot(1,0)
    qc.compile()

    assert np.allclose(qc.simulate(vec).state, expected)

    # Three qubit bell state construction
    qc = Circuit(3).h(0).cnot(0,2)
    vec = StateVector.from_basis_state('000')
    expected = 1 / np.sqrt(2) * np.array([1,0,0,0,0,1,0,0])
    qc.compile()

    assert np.allclose(qc.simulate(vec).state, expected)

    # Reversing qubits should produce same result
    qc = Circuit(3).h(2).cnot(2,0)
    qc.compile()

    assert np.allclose(qc.simulate(vec).state, expected)

    # Test GHZ state construction
    qc = Circuit(3).h(0).cnot(0,1).cnot(0,2)
    expected = 1 / np.sqrt(2) * np.array([1,0,0,0,0,0,0,1])
    qc.compile()

    assert np.allclose(qc.simulate(vec).state, expected)

    # Check measurement statistics
    num_samples = 1000
    results = qc.sample(num_samples, vec)
    assert len(results.keys()) == 2
    assert '111' in results.keys() and '000' in results.keys()
    assert abs(results['000'] / num_samples - 0.5) < 0.05
    assert abs(results['111'] / num_samples - 0.5) < 0.05

    # Changing circuit structure still produces the same results
    # (and note that there's weird permutations here)
    qc = Circuit(3).h(0).cnot(0,2).cnot(2,1)
    qc.compile()

    assert np.allclose(qc.simulate(vec).state, expected)

@pytest.mark.parametrize("seed", range(100))
def test_optimizer_correctness(seed: int):
    rng = default_rng(seed)

    # Generate a random circuit and copies
    qc = random_circuit(rng)
    optimized_qc = qc.copy()
    double_optimized_qc = qc.copy()

    # Compile one copy without optimization
    qc.validate()
    qc.lower()
    qc.compiled = True

    # Compile/optimize another copy once
    optimized_qc.compile()

    # Double compile/optimize another copy
    double_optimized_qc.compile()
    double_optimized_qc.compile()

    # Generate a few states
    num_qubits = qc.num_qubits
    states = [StateVector.random_state(num_qubits, rng) for _ in range(5)]

    # Simulating each state should produce the same result on each of the three circuits no matter how optimized
    for state in states:
        assert np.allclose(qc.simulate(state).state, optimized_qc.simulate(state).state)
        assert np.allclose(optimized_qc.simulate(state).state, double_optimized_qc.simulate(state).state)

def random_circuit(rng: Generator | None=None) -> Circuit:
    if rng is None:
        rng = default_rng()

    # Pool of gates to generate circuits from
    GATES = [
            (Operator.identity,1),
            (Operator.hadamard,1),
            (Operator.pauli_x,1),
            (Operator.pauli_y,1),
            (Operator.pauli_z,1),
            (Operator.rx,1),
            (Operator.ry,1),
            (Operator.rz,1),
            (Operator.cnot,2),
            (Operator.swap,2),
            (Operator.toffoli,3),
            (Operator.fredkin,3)
        ]

    # Choose how many qubits we're working with and filter the gate pool accordingly
    num_qubits = rng.choice(range(1,6))
    gates = [(gate, size) for gate, size in GATES if size <= num_qubits]

    qc = Circuit(num_qubits)

    # Set a number of gates
    num_gates = rng.choice(range(1,31))

    last_gate = None
    last_qubits = None

    # Generate gates
    for _ in range(num_gates):
        # 25% chance of repeating the last gate
        if last_gate is not None and rng.random() < 0.25:
            gate, size = last_gate

            # If repeating the last gate, 50% chance of it acting on the same qubits
            if last_qubits is not None and rng.random() < 0.5:
                qubits = last_qubits
            else:
                qubits = tuple(rng.choice(num_qubits, size=size, replace=False))

        # If we're not repeating a gate, pick a new one
        else:
            gate, size = gates[rng.integers(len(gates))]
            qubits = tuple(rng.choice(num_qubits, size=size, replace=False))

        # If we're doing a rotation, pick an angle
        theta = rng.uniform(0, 2*np.pi) if any(rotation in gate.__name__
                                            for rotation in ['rx','ry','rz']) else None

        # If we're doing an identity, that needs an extra parameter
        if_identity = 1 if 'identity' in gate.__name__ else None

        # Add the appropriate gate
        if theta is not None:
            qc._add_gate(gate(theta), qubits)
        elif if_identity is not None:
            qc._add_gate(gate(if_identity), qubits)
        else:
            qc._add_gate(gate(), qubits)

        # 10% chance that the generator hangs on to the last gate it used
        # (to try to generate X -> I -> X type of sequences)
        if rng.random() < 0.9:
            last_gate = (gate,size)
            last_qubits = qubits

    return qc