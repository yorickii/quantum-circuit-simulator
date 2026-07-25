# quantum-circuit-simulator
Simple package to simulate quantum circuits

## StateVector class

### measure_qubit

Explanation of weird bit manipulations:

Suppose I have a three qubit system (therefore eight basis states), and I want to measure the qubit indexed as 1--the second qubit, the middle one, however you want to call it. Then I'll need to know the probability that the middle qubit is `0` and the probability that it is `1`.

The usual ordering of basis states in quantum information matches binary: the first basis state is usually |00...00>, with as many zeros as needed. Then the next is |00...01>, the next is |00...10>, the next is |00...11>, etc. That is, to get the bit representation of any component of any state vector, we just take the index of that component.

Next, note that in python the `&` operator is a bitwise and; that is, `101 & 001` returns `001`. We just want to identify if any given bit is a zero or one, so if we can get the correct bit and `&` it with `1`, we will get `1` if the bit was `1` and `0` if the bit was `0`.

So how do you get the right bit to do the `&` with? Shift the entire index bitstring to the right by some amount. If I want to know about the rightmost qubit, I shift by nothing; if I want to know about the leftmost qubit, I shift by the maximum amount. The maximum amount you can shift without completely discarding your original bitstring is one less than the number of bits. Therefore, we shift by `num_qubits - 1 - qubit` bits. This is zero if we are measuring the rightmost qubit (`qubit == num_qubits - 1`) and `num_qubits - 1` if we are measuring the leftmost (`qubit == 0`). In our example from earlier, we would have `num_qubits == 3` and `qubit == 1`, and would thus shift by one bit.

What about the leftover leading bits? They don't matter. If I have a two (or more) bit bitstring and I `&` with a one bit bitstring, python just left-pads the one bit bitstring with enough zeros to make the operation work. Since bitwise and only returns `1` if both arguments are `1`, everything's fine.

## Circuit class

### compile

Explanation of weird bit manipulations:

First I'll explain the calculation to get the `bits` variable.

We start with a list of every index in the state vector. Consider each index as its bitstring; then each index gives the qubit ordering for that component of the state vector. We want to permute the ordering. Let's start by isolating each bit from the bitstring. We can do this by bitshifting the bitstring bit by bit and `&`ing with `1`; this will give `1` if the rightmost bit of the bitstring is `1` and `0` otherwise. If we pass a list to the bitshift operator, it will give us a list of bitshifted numbers out. So, for instance, if I did `[4] >> [0, 1, 2]` I would get `[4, 2, 1]` out. If I did the same operation and then `&`ed with `1`, I would get `[0, 0, 1]`. Note that this is nearly the bitstring for 4.

So what amount do we bitshift by? If we just want to get the original bitstring, we would shift by `(num_qubits - 1 - arange(num_qubits))`. For a four qubit system, this gives `[3, 2, 1, 0]`. Looking at our previous example, `[4] >> [3, 2, 1, 0] gives [0, 1, 2, 4]` and `&`ing with `1` gives `[0, 1, 0, 0]` which we can read as `0100` or 4, in a four digit binary representation.

Now, we don't just want the bit representations&mdash;we want to permute them. So if we permute the order we do the bitshifts in, we will end up with permuted bitstrings. If we have a three qubit system and want to place the rightmost qubit first, then the first bitshift we want to do is zero, or no shift. That's `num_qubits - 1 - 2`. If we want to put the leftmost qubit in the middle, we want to do two bitshifts next: `num_qubits - 1 - 0`. Finally, if we want the middle qubit last, we want to do one bitshift, which is `num_qubits - 1 - 1`. That is, we want the list we give to our bitshift calculation to be `[2, 0, 1]`. This is exactly the permuted list of qubit indices calculated in `permuted_qubits`.

To reinterpret these lists of bits (in `bits`) as state vector indices, we do the bit manipulations in `permuted_state_indices`. Each list of bits needs to somehow be combined back into one bitstring. We can do this by shifting each bit in the list back to the left the correct number of places and then summing each list of bits (separately from each other, of course). The result of the sum will be the correct bitstring, which will of course give the correct index.

The amount by which to bitshift is dictated by the order in the list. If we have a three qubit system, the first element in the list of qubits needs to end up in the fours place, which means it needs to be shifted twice. That's `num_qubits - 1 - 0`. The second element needs to end up in the twos place, which means a shift of one&mdash;`num_qubits - 1 - 1`. And the last element needs to be in the ones place, which is of course no shift at all, or `num_qubits - 1 - 2`. That is, the list we give to the bitshift calculation is `[0, 1, 2]`; hence we do `bits << (num_qubits - 1 - arange(num_qubits))`. For the list of bits giving `[4]`, we would have (after bitshifts) `[0000, 100, 00, 0]`. Summing gives `0100`, or 4. So after summing along the horizontal axis, we have a permuted list of state vector indices.