# quantum-circuit-simulator
Simple package to simulate quantum circuits

## StateVector class

### measure_qubit

Explanation of weird bit manipulations:

Suppose I have a three qubit system (therefore eight basis states), and I want to measure the qubit indexed as 1--the second qubit, the middle one, however you want to call it. Then I'll need to know the probability that the middle qubit is 0 and the probability that it is 1.

The usual ordering of basis states in quantum information matches binary: the first basis state is usually |00...00>, with as many zeros as needed. Then the next is |00...01>, the next is |00...10>, the next is |00...11>, etc. That is, to get the bit representation of any component of any state vector, we just take the index of that component.

Next, note that in python the & operator is a bitwise and; that is, 101 & 001 returns 001. We just want to identify if any given bit is a zero or one, so if we can get the right bit and bitwise and it with 1, we will get 1 if the bit was 1 and 0 if the bit was 0.

So how do you get the right bit to do the and with? Shift the entire index bitstring to the right by some amount. If I want to know about the rightmost qubit, I shift by nothing; if I want to know about the leftmost qubit, I shift by the maximum amount. The maximum amount you can shift without completely discarding your original bitstring is one less than the number of bits. Therefore, we shit by num_qubits - 1 - qubit bits. This is zero if we are measuring the rightmost qubit (qubit = num_qubits - 1) and num_qubits - 1 if we are measuring the leftmost (qubit = 0). In our example from earlier, we would have num_qubits = 3 and qubit = 1, and would thus shift by 1 bit.

What about the leftover leading bits? They don't matter. If I have a two (or more) bit bitstring and I bitwise and with a one bit bitstring, python just left-pads the one bit bitstring with enough zeros to make the operation work. Since bitwise and only returns 1 if both arguments are 1, everything's fine.