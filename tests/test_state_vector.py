from state_vector import StateVector, invert_permutation
from operators import Operator
import numpy as np
from numpy.linalg import norm
from numpy.random import randint, choice
from scipy.stats import chisquare
import pytest
import re

def test_init():
    # Test single qubit init
    vec = StateVector(1)
    assert np.allclose(vec.state, np.array([1,0]))

    # Test multi qubit init
    vec = StateVector(3)
    assert np.allclose(vec.state, np.array([1,0,0,0,0,0,0,0]))

    # Test rejection of 0 qubit states
    with pytest.raises(ValueError, match="State must contain at least one qubit."):
        StateVector(0)

    # Test rejection of dimension mismatch
    with pytest.raises(ValueError, match=re.escape("Must provide a state vector of length 2**1")):
        StateVector(1, np.array([1,0,0]))

def test_normalization():
    # Test that arrays that aren't normalized get normalized when constructed into states
    not_normal = np.array([2,0])
    vec = StateVector(1, not_normal)

    assert not np.allclose(vec.state, not_normal)
    assert np.allclose(vec.state, np.array([1,0]))

    # Test rejection of non normalizable states
    with pytest.raises(ValueError, match=re.escape("State vector is non-normalizable. State: [0 0]. Norm: 0.0")):
        StateVector(1, np.array([0,0]))

def test_from_index():
    # Test that from_index works correctly on valid indices
    num_qubits = randint(1,11)
    index = randint(1, 2**num_qubits)

    vec = StateVector.from_index(index, num_qubits)
    comparison = np.zeros_like(vec.state)
    comparison[index] = 1

    assert np.allclose(vec.state, comparison)

    # Test that from_index correctly rejects positive indices that are too large
    with pytest.raises(ValueError) as info:
        StateVector.from_index(2**num_qubits + 1, num_qubits)

    assert str(info.value) == f"Index out of range. Index: {2**num_qubits + 1}. State vector length: {2**num_qubits}"

    # Test that from_index works correctly on negative indices that are in range
    index = randint(-2**num_qubits, -1)

    vec = StateVector.from_index(index, num_qubits)
    comparison = np.zeros_like(vec.state)
    comparison[index] = 1

    assert np.allclose(vec.state, comparison)

    # Test that from_index correctly rejects negative indices which are too negative
    with pytest.raises(ValueError) as info:
        StateVector.from_index(-2**num_qubits - 1, num_qubits)

    assert str(info.value) == (f"Index out of range. Index: -1. State vector length: {2**num_qubits}")

def test_from_basis_state():
    # Test that from_basis_state correctly initializes from a valid bitstring
    length = randint(1, 4)
    bitstring = ''.join(choice(['0', '1']) for _ in range(length))
    index = int(bitstring, 2)

    vec = StateVector.from_basis_state(bitstring)
    comparison = np.zeros_like(vec.state)
    comparison[index] = 1

    assert np.allclose(vec.state, comparison)

    # Test that from_basis_state correctly rejects invalid bitstrings
    with pytest.raises(ValueError, match="Basis state must be a string of 0s and 1s"):
        StateVector.from_basis_state('102')

def test_from_qubits():
    # Test that from_qubits correctly initializes from a valid list of qubit factors
    single_qubit = np.array([randint(0, 10), randint(0, 10)])
    magnitude = norm(single_qubit)

    factors_for_func = ['up', 0, 'down', 1, single_qubit / magnitude]
    factors_for_comp = [np.array([1,0]), np.array([1,0]), np.array([0,1]),
                        np.array([0,1]), single_qubit / magnitude]

    vec = StateVector.from_qubits(factors_for_func)

    comparison = factors_for_comp[0]
    for factor in factors_for_comp[1:]:
        comparison = np.kron(comparison, factor)

    assert np.allclose(vec.state, comparison)

    # Test that from_qubits correctly normalizes non-normal factors
    factors_for_func[-1] = single_qubit
    vec = StateVector.from_qubits(factors_for_func)

    assert np.allclose(vec.state, comparison)

    # Test that from_qubits rejects non-normalizable factors
    factors_for_func[-1] = np.array([0,0])

    with pytest.raises(ValueError) as info:
        StateVector.from_qubits(factors_for_func)

    assert str(info.value) == "Cannot normalize a zero vector. Qubit: [0 0]. Norm: 0.0"

    # Test that from_qubits correctly rejects unrecognized qubit labels
    factors_for_func[-1] = 'right'

    with pytest.raises(ValueError) as info:
        StateVector.from_qubits(factors_for_func)

    assert str(info.value) == "Unrecognized state label: right"

    factors_for_func[-1] = 2
    
    with pytest.raises(ValueError) as info:
        StateVector.from_qubits(factors_for_func)

    assert str(info.value) == "Unrecognized state label: 2"

    # Test that from_qubits correctly rejects qubit factors with the wrong shape
    factors_for_func[-1] = np.array([1,0,0])
        
    with pytest.raises(ValueError) as info:
        StateVector.from_qubits(factors_for_func)

    assert str(info.value) == "Single-qubit state vectors must have shape (2,), got (3,)"

    # Test that from_qubits rejects qubit types it doesn't know
    factors_for_func[-1] = 0.123

    with pytest.raises(ValueError) as info:
        StateVector.from_qubits(factors_for_func)

    assert str(info.value) == "Invalid state label type: <class 'float'>"

def test_random_basis():
    # Test that random_basis is replicable with a fixed seed
    seed = randint(100)
    num_qubits = randint(1,4)

    vec = StateVector.random_basis(num_qubits, seed)
    vec_2 = StateVector.random_basis(num_qubits, seed)

    assert np.allclose(vec.state, vec_2.state)

    # Test that random_basis only sets one component nonzero
    how_many = 0
    for component in vec.state:
        how_many += 1 if component != 0 else 0

    assert how_many == 1

    # Test that random_basis generates a normalized state
    magnitude = norm(vec.state)

    assert np.isclose(1, magnitude)

def test_random_state():
    # Test that random_basis is replicable with a fixed seed
    seed = randint(100)
    num_qubits = randint(1,4)

    vec = StateVector.random_state(num_qubits, seed)
    vec_2 = StateVector.random_state(num_qubits, seed)

    assert np.allclose(vec.state, vec_2.state)

    # Test that random_basis generates a normalized state
    magnitude = norm(vec.state)

    assert np.isclose(1, magnitude)

def test_uniform_state():
    # Test that uniform_state generates a uniform state
    num_qubits = randint(1,4)

    vec = StateVector.uniform_state(num_qubits)

    is_equal = True
    for component in vec.state[1:]:
        is_equal = False if not np.isclose(vec.state[0], component) else True

    assert is_equal

    # Test that uniform_state generates normalized states
    magnitude = norm(vec.state)

    assert np.isclose(1, magnitude)

def test_evolve():
    # Test that evolving by an identity doesn't change anything
    vec = StateVector(3)
    id = Operator.identity(3)

    new_vec = vec.evolve(id, in_place=False)

    assert np.allclose(vec.state, new_vec)

    # Test that Hadamard and Pauli evolutions produce expected results
    up = StateVector(1)
    down = StateVector(1, np.array([0,1]))
    h = Operator.hadamard()

    assert np.allclose(up.evolve(h, in_place=False), 1/np.sqrt(2) * np.array([1,1]))
    assert np.allclose(down.evolve(h, in_place=False), 1/np.sqrt(2) * np.array([1,-1]))

    x = Operator.pauli_x()

    assert np.allclose(up.evolve(x, in_place=False), down.state)
    assert np.allclose(down.evolve(x, in_place=False), up.state)

    y = Operator.pauli_y()

    assert np.allclose(up.evolve(y, in_place=False), 1j * down.state)
    assert np.allclose(down.evolve(y, in_place=False), -1j * up.state)

    z = Operator.pauli_z()

    assert np.allclose(up.evolve(z, in_place=False), up.state)
    assert np.allclose(down.evolve(z, in_place=False), -down.state)

    # Test that evolve rejects dimension mismatches
    with pytest.raises(ValueError) as info:
        up.evolve(id, in_place=False)

    assert str(info.value) == "Operator shape (8, 8) incompatible with 1 qubits."

    # Test that in_place works correctly
    up.evolve(h)

    assert np.allclose(up.state, 1/np.sqrt(2) * np.array([1,1]))

def test_measure_state():
    num_tests = 1000
    vec = StateVector.random_state(3)

    measurements = [0 for _ in range(len(vec.state))]

    for _ in range(num_tests):
        bitstring, result = vec.measure_state(in_place=False)

        # Test that the returned bitstring is always a bitstring
        for bit in bitstring:
            assert bit in '01'

        # Test that the bitstring gives a valid index
        index = int(bitstring, 2)
        assert index < len(vec.state)

        # Test that only one value is equal to 1 and that it is the value indexed by the returned bitstring
        how_many = 0
        collapsed_index = -100
        for i, component in enumerate(result):
            how_many += component
            if component == 1:
                collapsed_index = i

        assert np.isclose(how_many, 1)
        assert collapsed_index == index

        measurements[index] += 1

    # Test that the measurement frequencies correspond with the probabilities calculated from the state vector
    result = chisquare(np.array(measurements), f_exp=abs(vec.state)**2 * num_tests)
    assert result.pvalue > 0.001

    # Test that measuring a basis state always gives the basis state we started with
    vec_1 = StateVector.from_basis_state('101')

    for _ in range(num_tests):
        bitstring, result = vec_1.measure_state(in_place=False)

        assert bitstring == '101'
        for i, component in enumerate(result):
            if i == int(bitstring, 2):
                assert component == 1
            else:
                assert component == 0

    # Test that seeds work
    bitstring_1, result_1 = vec.measure_state(seed=1234, in_place=False)
    bitstring_2, result_2 = vec.measure_state(seed=1234, in_place=False)

    assert bitstring_1 == bitstring_2 and np.allclose(result_1, result_2)

    # Test that in place measurement works
    bitstring, result = vec.measure_state()

    assert np.allclose(result, vec.state)

def test_measure_qubit():
    num_tests = 1000
    vec = StateVector.random_state(2)

    measurements = [0, 0]

    for _ in range(num_tests):
        measurement, new_state = vec.measure_qubit(1, in_place=False)
        measurements[measurement] += 1

        # Test that state vector correctly collapses
        if measurement == 0:
            assert new_state[1] == 0 and new_state[3] == 0
            assert new_state[0] != 0 and new_state[2] != 0
        else:
            assert new_state[0] == 0 and new_state[2] == 0
            assert new_state[1] != 0 and new_state[3] != 0

        # Check for normalization
        assert np.isclose(sum(abs(new_state)**2), 1)

    # Test that measured frequencies match predicted ones
    probs = np.array([abs(vec.state[0])**2 + abs(vec.state[2])**2, abs(vec.state[1])**2 + abs(vec.state[3])**2])

    result = chisquare(np.array(measurements), f_exp=probs * num_tests)
    assert result.pvalue > 0.001

    # Check for out-of-range qubit index rejection
    with pytest.raises(ValueError) as info:
        vec.measure_qubit(2, in_place=False)

    assert str(info.value) == "Qubit index out of range. Number of qubits: 2. Qubit index: 2."

    # Check that seeds work
    measurement_1, new_state_1 = vec.measure_qubit(1, seed=1234, in_place=False)
    measurement_2, new_state_2 = vec.measure_qubit(1, seed=1234, in_place=False)

    assert measurement_1 == measurement_2 and np.allclose(new_state_1, new_state_2)

    # Check that in place measurement works
    measurement, new_state = vec.measure_qubit(1)

    assert np.allclose(new_state, vec.state)

def test_permutations():
    vec = StateVector.random_state(2)

    permutation = [2,0,3,1]
    inverse = invert_permutation(permutation)

    permuted = StateVector(2, vec.permute_state(permutation, in_place=False))

    # Make sure components end up where expected
    for i, j in enumerate(permutation):
        assert np.isclose(permuted.state[j], vec.state[i])

    # Make sure inverse permutations work
    inverted = permuted.permute_state(inverse, in_place=False)

    assert np.allclose(inverted, vec.state)