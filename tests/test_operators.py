from state_vector import StateVector
from operators import Operator
import numpy as np
import pytest

def test_init():
    # Test default initialization
    op = Operator(1)

    assert np.allclose(op.operator, np.eye(2))

    # Test that different qubit numbers give correct dimensions
    assert Operator(2).operator.shape == (4, 4)
    assert Operator(3).operator.shape == (8, 8)
    assert Operator(4).operator.shape == (16, 16)

    # Test that bad dimensions are rejected
    with pytest.raises(ValueError) as info:
        Operator(1, np.array([[1,2,3],[4,5,6]]))

    assert str(info.value) == "Must provide an operator of shape (2**1,2**1)"

def test_dagger():
    # Test that dagger produces the expected results
    op = Operator.hadamard()
    dagger = op.dagger()

    assert np.allclose(dagger.operator, op.operator.T)
    assert dagger.symbol == "H\u2020"

    op = Operator.pauli_y()
    dagger = op.dagger()

    assert np.allclose(op.operator, dagger.operator)
    assert dagger.symbol == "Y\u2020"

def test_mult():
    # Check that operator multiplication works as expected
    op_1 = Operator.hadamard()
    op_2 = Operator.pauli_x()
    op_3 = op_1 @ op_2

    assert np.allclose(op_3.operator, op_1.operator @ op_2.operator)

    # Make sure operator multiplication rejects dimension mismatch
    op_4 = Operator.cnot()

    with pytest.raises(ValueError) as info:
        op_1 @ op_4

    assert str(info.value) == "Shape mismatch: 1 != 2"

    # Check that operators can multiply states
    vec = StateVector.from_basis_state('0')
    new_vec = op_3 @ vec

    assert np.allclose(new_vec.state, op_3.operator @ vec.state)

def test_tensor():
    # Test that tensor products produce expected results
    op = Operator.hadamard()
    new_op = Operator(1).tensor(op)
    assert np.allclose(new_op.operator, np.block([[op.operator, np.zeros((2,2))],
                                                  [np.zeros((2,2)), op.operator]]))
    assert new_op.num_qubits == 2

    new_new_op = Operator(2).tensor(new_op)
    assert np.allclose(new_new_op.operator,
                       np.block([[new_op.operator, np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))],
                                 [np.zeros((4,4)), new_op.operator, np.zeros((4,4)), np.zeros((4,4))],
                                 [np.zeros((4,4)), np.zeros((4,4)), new_op.operator, np.zeros((4,4))],
                                 [np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), new_op.operator]]))
    assert new_new_op.num_qubits == 4

    # Test that ^ override works
    assert np.allclose(new_op.operator, (Operator(1) ^ op).operator)

    # Test that from_factors agrees with tensor products
    assert np.allclose(Operator.from_factors(["i", "h"]).operator, new_op.operator)

    # Test that from_factors accepts supported factor types
    assert np.allclose(Operator.from_factors([np.eye(2), "h"]).operator, new_op.operator)
    assert np.allclose(Operator.from_factors([Operator(1), Operator.hadamard()]).operator, new_op.operator)

    # Test that from_factors rejects bad factors
    with pytest.raises(ValueError, match="Unrecognized operator name: operator"):
        Operator.from_factors(["i", "operator"])

    with pytest.raises(ValueError, match="Expected a single-qubit operator, got a 2 qubit operator"):
        Operator.from_factors([Operator(2), "h"])

    with pytest.raises(ValueError, match="Invalid type <class 'float'> for an operator factor."):
        Operator.from_factors([0.1234, "h"])