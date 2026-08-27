# quantum-circuit-simulator
This is a simple package to simulate quantum circuits, built as a personal project to help me learn more about quantum computing and build skills in that field. At the moment, it is not particularly optimized, since I have mostly concerned myself with overall function as opposed to high performance.

This documentation is largely written for my own benefit, to help me remember all of the elements of this software&mdash;I don't really have a lot of experience writing documentation, so I don't know all of the conventions to follow. Hopefully this is easy enough for others to follow.

### Qubit convention

I use the convention where basis states are ordered like

$$
    \ket{00}, \ket{01}, \ket{10}, \ket{11};
$$

that is, the $n^{\text{th}}$ basis state is given by the binary representation of $n$. I consider the qubit on the left to be qubit zero.

I'm pretty sure this convention isn't standard in any "real" QC packages. My primary motivation for using it is that it made sense to me. I'm sure there are very compelling reasons why some other convention should be used.

### AI disclosure

Since my goal with this project was to teach myself more about quantum computing, it wouldn't make much sense for the whole thing to be vibe coded. That said, I did make some use of AI. I would estimate that about 80% of the code in this package is human written. My primary uses for AI are:

- validating broad structural decisions before committing too deeply to them
- seeking cleaner ways of accomplishing something after finding a solution myself
- checking my implementations for completeness (am I missing standard functionality)

Very rarely do I copy large sections of AI generated code wholesale; it is generally at most a line or two. None of this README was generated in any way by AI. The em dashes are my own :)

At the time of writing, the only parts of code that I can remember AI contributing significantly to (as in, I did not have any solution myself before seeing the AI generated solution) are the bit manipulations for qubit measurements and state vector permutations. Bit-level programming is not a strong suit of mine, and something I struggle to think in terms of. That said, the explanations of the bit manipulations found later in this readme are entirely my own, since I wanted to understand what the code was doing. While I did not come up with the operations myself, seeing them in context has helped me understand bit-level programming better&mdash;I'll note that after implementing qubit measurements I got much further on my own with permutations.

## Package overview

This simulator is divided (so far) into three main classes + one dataclass. We have:

- StateVector: does what it says on the tin. Has lots of specialized constructors for particular state configurations.
- Operator: likewise, including the constructors.
- Circuit: contains all the simulation logic, as well as methods for compiling/optimizing the set of gates it's given.
- GateApplication: a small dataclass that stores an operator and the qubits the operator is to act on.

In principle, if all you want to do is initialize a state and simulate its interaction with a circuit, you only need the StateVector class and the Circuit class. Operators and GateApplications are handled internally. Of course if you want to look more closely at the internal logic, then you might be interested in those other classes.

A simple example might look something like this:

```python
from circuit import Circuit

qc = Circuit(3).h(0).h(2).cnot(2,0)

result = qc.sample(1000)

print(result)

-----------------------------------

>> {np.str_('000'): 246, np.str_('100'): 260, np.str_('001'): 266, np.str_('101'): 228}
```

This constructs a 3 qubit circuit with a Hadamard on the first and third qubits, and then a CNOT with the third qubit as control and the first qubit as the target. Then we run a state&mdash;`Circuit.sample()` defaults to the all-up state, so here we don't even need the StateVector class&mdash;through the circuit and measure the final state 1000 times, and return a dictionary mapping basis vectors to counts, from which we could get probabilities.

There are of course many other gates that are natively supported, and the user can also construct any gate they desire.

Do also note that this package supports pretty printing of circuits:

```python
print(qc)

-----------------------------------

q0 ───H───────X───
              │   
q1 ───────────┼───
              │   
q2 ───────H───■───
```

### Tests

I would not describe myself as a software engineer, so I unfortnately didn't think to be maintaining tests as I developed. This is something I intend to go back and rectify, but until that point this note will stay here in case anyone wonders where the tests are.

I *have* tested things... I just didn't save them or structure them in an easily replicable way. Oops.

## StateVector class

For those with any exposure to quantum computing or quanum mechanics, this class should be largely self-documenting. It stores a complex numpy array as the components of the state vector, along with the number of qubits the state vector represents.

Let's run down the member functions.

### \_\_init\_\_

Takes an integer `num_qubits` and an optional complex numpy array. If no array is provided, initializes to the all up state. Verifies the initialized state is normalized.

### from_index

Takes an integer `index`, and an integer `num_qubits`. Initializes a state vector of an appropriate length with only the specified element nonzero. Thus, the index must be less than $2^{num\_ qubits}$.

### from_basis_state

Functionally the same as `from_index`. Takes a bitstring (specifying a basis state), converts the bitstring to an index and calculates the number of qubits as the length of the bitstring, and then calls `from_index`.

### random_basis

Randomly picks a basis state and calls `from_index`. Also accepts a seed for random operations.

### from_qubits

Takes a list of qubits and constructs a state vector from the tensor product. Qubits can be provided as "up"/"down", 0/1, or a full one-qubit state vector (a two element complex array).

### random_state

Constructs a fully random state having the specified number of qubits. Also accepts a seed for random operations.

### uniform_state

Takes an integer `num_qubits` and returns a state vector with equal amplitude on each component.

### evolve

Takes an `Operator` object, multiplies the state vector by the operator, and returns the result. If the boolean flag `in_place` is set to `True` (as it is by default), then this return is probably unnecessary.

### probabilities

Returns a dictionary mapping basis state bitstrings to the absolute value squared of the amplitude associated with that basis state.

### probability

Returns the absolute value squared of the amplitude of the specified component. Components can be specified either as an index or a bitstring.

### sample

Accepts an integer `num_samples`, and returns a dictionary mapping bitstrings to the number of times that basis state was observed in `num_samples` tests. Also accepts a seed for random operations.

### measure_state

Returns a basis state randomly selected from the state vector (weighted by amplitudes)&mdash;that is, simulates measuring all qubits at once. Accepts a seed for random operations and a boolean flag to specify whether you want the operation to happen in place or not.

### measure_qubit

Measures the state of the qubit indexed by the parameter `qubit`. That is, weighted randomly chooses whether the specified qubit is up or down, and collapses the state vector accordingly. Also accepts a seed and an `in_place` flag.

As noted in the code, here is an extended explanation of the bit manipulation logic:

Suppose I have a three qubit system (therefore eight basis states), and I want to measure the qubit indexed as 1--the second qubit, the middle one, however you want to call it. Then I'll need to know the probability that the middle qubit is `0` and the probability that it is `1`.

The ordering of basis states in my adopted convention matches binary: the first basis state is usually $\ket{00...00}$, with as many zeros as needed. Then the next is $\ket{00...01}$, the next is $\ket{00...10}$, the next is $\ket{00...11}$, etc. That is, to get the bit representation of any component of any state vector, we just take the index of that component.

Next, note that in python the `&` operator is a bitwise and; that is, `101 & 001` returns `001`. We just want to identify if any given bit is a zero or one, so if we can get the correct bit and `&` it with `1`, we will get `1` if the bit was `1` and `0` if the bit was `0`.

So how do you get the right bit to do the `&` with? Shift the entire index bitstring to the right by some amount. If I want to know about the rightmost qubit, I shift by nothing; if I want to know about the leftmost qubit, I shift by the maximum amount. The maximum amount you can shift without completely discarding your original bitstring is one less than the number of bits. Therefore, we shift by `num_qubits - 1 - qubit` bits. This is zero if we are measuring the rightmost qubit (`qubit == num_qubits - 1`) and `num_qubits - 1` if we are measuring the leftmost (`qubit == 0`). In our example from earlier, we would have `num_qubits == 3` and `qubit == 1`, and would thus shift by one bit.

What about the leftover leading bits? They don't matter. If I have a two (or more) bit bitstring and I `&` with a one bit bitstring, python just left-pads the one bit bitstring with enough zeros to make the operation work. Since bitwise and only returns `1` if both arguments are `1`, everything's fine.

### copy

Returns a deep copy of a `StateVector` object.

### permute_state

Accepts a vector specifying a new order for the components of the state and returns the result of permuting the state vector in that way. This is explicitly only permuting the state vector&mdash;there's no qubit awareness at this level.

### \_\_matmul\_\_

Overwrites the `@` operator to enable matrix multiplication between `StateVector` objects. Specifically, takes the complex conjugate of `self` and multiplies by `other`.

### \_\_str\_\_

Pretty print the state vector. Should look something like:

```python
|00>: 0.5
|01>: 0.5
|10>: 0.5
|11>: 0.5
```

Ignores any components which are close to zero.

## Operator class

As with the StateVector class, this class is probably almost entirely self documenting to those already familiar with quantum computing. It stores the actual matrix representation of the operator, as well as the number of qubits the operator acts on and a string `symbol` which is primarily used for printing circuits (but also for a couple of other things).

Once again, we'll run down the member functions.

### \_\_init\_\_

Takes an integer `num_qubits`, an optional complex array representation of the operator, and an optional string `symbol`. If no complex array is provided, the operator is initialized to the identity of size $2^{num\_ qubits}$. If no `symbol` is provided, the operator is labeled as `'O'`.

### identity

Initializes an identity operator to act on the specified `num_qubits`.

### hadamard

Initializes a Hadamard operator:

$$
\frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \\ \end{pmatrix}.
$$

### pauli_x

Initializes a Pauli X operator:

$$
\begin{pmatrix} 0 & 1 \\ 1 & 0 \\ \end{pmatrix}.
$$

### pauli_y

Initializes a Pauli Y operator:

$$
\begin{pmatrix} 0 & -i \\ i & 0 \\ \end{pmatrix}.
$$

### pauli_z

Initializes a Pauli Z operator:

$$
\begin{pmatrix} 1 & 0 \\ 0 & -1 \\ \end{pmatrix}.
$$

### phase

Initializes a phase operator:

$$
\begin{pmatrix} 1 & 0 \\ 0 & i \\ \end{pmatrix}.
$$

### t

Initializes a T operator:

$$
\begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \\ \end{pmatrix}.
$$

### rx

Initializes an X rotation operator using angle `theta`:

$$
\begin{pmatrix} \cos \frac{\theta}{2} & -i \sin \frac{\theta}{2} \\ -i \sin \frac{\theta}{2} & \cos \frac{\theta}{2} \\ \end{pmatrix}.
$$

### ry

Initializes a Y rotation operator using angle `theta`:

$$
\begin{pmatrix} \cos \frac{\theta}{2} & -\sin \frac{\theta}{2} \\ \sin \frac{\theta}{2} & \cos \frac{\theta}{2} \\ \end{pmatrix}.
$$

### rz

Initializes a Z rotation operator using angle `theta`:

$$
\begin{pmatrix} e^{-i \theta / 2} & 0 \\ 0 & e^{i \theta / 2} \\ \end{pmatrix}.
$$

### phase_shift

Initializes a phase shift operator using angle `phi`:

$$
\begin{pmatrix} 1 & 0 \\ 0 & e^{i \phi} \\ \end{pmatrix}.
$$

### u

Initializes an arbitrary U operator from angles `theta`, `phi`, and `lam`bda:

$$
\begin{pmatrix} \cos \frac{\theta}{2} & -e^{i \lambda} \sin \frac{\theta}{2} \\ e^{i \phi} \sin \frac{\theta}{2} & e^{i (\phi + \lambda)} \cos \frac{\theta}{2} \\ \end{pmatrix}.
$$

### cnot

Initializes a CNOT operator (which is just a controlled Pauli X operator):

$$
\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ \end{pmatrix}.
$$

### cz

Initializes a controlled Pauli Z operator:

$$
\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \\ \end{pmatrix}.
$$

### cy

Initializes a controlled Pauli Y operator:

$$
\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & -i \\ 0 & 0 & i & 0 \\ \end{pmatrix}.
$$

### swap

Initializes a swap operator:

$$
\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ \end{pmatrix}.
$$

### controlled

Initializes a controlled version of `operator`. If `operator` has a `symbol`, the `symbol` for the new operator is just `'C'` plus the old `symbol`. Controlled operators follow the structure

$$
\begin{pmatrix} I & 0 \\ 0 & \widehat{O} \\ \end{pmatrix},
$$

where $\widehat{O}$ represents the operator that is to be controlled, and the dimensions of $I$ and $0$ are the same as the dimensions of $\widehat{O}$.

### toffoli

Initializes a Toffoli operator (which is a controlled CNOT operator):

$$
\begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ \end{pmatrix}.
$$

### fredkin

Initializes a Fredkin operator (which is a controlled swap operator):

$$
\begin{pmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ \end{pmatrix}.
$$

### from_factors

Initializes a multi-qubit operator from a list of single-qubit factors. The factors can be given as complex numpy arrays, `Operator` objects, or strings&mdash;`'hadamard'`, `'h'`, `'x'`, `'y'`, `'z'`, `'phase'`, `'s'`, `'t'`, and `'i'` are all supported. Computes the tensor product of the provided factors (in the order provided), and computes the resulting `num_qubits`. Also accepts an optional `symbol` parameter; defauls to `O` if none is provided.

At this point I don't remember why I needed `from_factors` to be explicitly single-qubit factors, but I'm pretty sure there was a good reason. Might just be because I wanted it to dynamically figure out how many qubits?... but I don't think that would be an insurmountable issue with multi-qubit factors. So I'm not sure.

### \_from\_factors

Functionally the same as `from_factors`, but does not verify that provided factors are single-qubit. Meant for internal use, mostly.

### tensor

Compute the tensor product between two `Operator` objects.

### dagger

Returns the complex conjugate of an `Operator`. Modifies the initial operator's symbol with the addition of a &#8224; symbol.

### set_symbol

Modifies an operator's symbolic representation to the provided one.

### \_\_matmul\_\_

Overwrites the operator `@` to be able to multiply `Operators` with other `Operators` and also with `StateVectors`.

### \_\_mul\_\_

Overwrites the operator `*` to enable scalar multiplication of `Operator` objects.

### \_\_xor\_\_

Overwrites the operator `^` to call `Operator.tensor()` when used on `Operator` objects.

## \_\_str\_\_

Returns the string output of the `Operator`'s matrix representation.

## Circuit class

The `Circuit` class could perhaps be thought of as a fancy container for a bunch of `Operators`. Of course the reality is a little more complex than that, but at a fundamental level that is pretty much what's going on: there's a bunch of `Operators` in some sequence, and we apply them successively to some quantum state so that we can then measure the output and learn something from it.

### \_\_init\_\_

Takes a `num_qubits` and an optional list of `GateApplications`. Stores a list of `GateApplications`, the `num_qubits`, a list `instructions` of `Operators` and permutation vectors, and a boolean flag indicating whether the circuit has been compiled.

### run

Takes an optional `initial_state` state vector. If no state vector is provided, uses the all up state and applies all gates in the circuit to the state. Also compiles the circuit if it's not already compiled. Returns the final state vector.

### sample

Runs an optional `initial_state` through the circuit and then gets `num_samples` from the result using `StateVector.sample()`. Also accepts a seed for random operations. Returns a dictionary mapping basis bitstrings to the number of times that basis was observed.

### simulate

Accepts an `initial_state` and applies all the operations in `instructions`. Returns the result.

### compile

Validates all `GateApplications`, removes unnecessary gates, and then generates the full-state `Operators` and permutation vectors to put in the list of `instructions`. Sets `compiled` to true.

### validate

Validates the list of `GateApplications`. At this stage, just checks that all qubits the `GateApplication` is supposed to act on are actually in range&mdash;that there aren't any negative indices or indices greater than the pre-specified number of qubits.

### optimize

Removes unnecessary gates. Specifically, replaces multiplications of involutary gates (e.g. $H H$) with identities, and combines same-axis rotation gates. At this stage only handles known involutary gates&mdash;does not check arbitrary gates for involutary-ness. Also only combines rotation gates that are labeled as such.

Note that in order for multi-qubit involutary gates to be canceled, they must be indexed identically to each other. That is, if there is a Toffoli gate acting on qubits (2,0,3), where the first two qubits listed are the control qubits and the last is the target, and another acting on (0,2,3), those will not cancel even though they technically could. This was just too complicated for me to feel like programming. Maybe I'll go back and add it at some point, but it doesn't feel that important at this stage.

### lower

Turns a list of `GateApplications` into a list of instructions&mdash;`Operators` and permutation vectors. The `Operators` constructed are the result of tensor products of factors: we add as many identities as needed and in the proper order, then call `Operator._from_factors` to construct the full state operator.

Permutation vectors are used any time there is a multi-qubit gate acting on non-adjacent qubits; e.g. a CNOT gate acting on qubits 0 and 2. To apply this to an actual state, we would need to permute the state such that qubits 0 and 2 are next to each other, since we can't split the CNOT gate apart. Note that for every permutation we do, we need to do an inverse permutation to get back to where we started.

As noted in the code, there are more interesting bit operations needed here to figure out the right permutations. Here is the extended explanation:

First I'll explain the calculation to get the `bits` variable.

We start with a list of every index in the state vector. Consider each index as its bitstring; then each index gives the qubit ordering for that component of the state vector. We want to permute the ordering. Let's start by isolating each bit from the bitstring. We can do this by bitshifting the bitstring bit by bit and `&`ing with `1`; this will give `1` if the rightmost bit of the bitstring is `1` and `0` otherwise. If we pass a list to the bitshift operator, it will give us a list of bitshifted numbers out. So, for instance, if I did `[4] >> [0, 1, 2]` I would get `[4, 2, 1]` out. If I did the same operation and then `&`ed with `1`, I would get `[0, 0, 1]`. Note that this is nearly the bitstring for 4.

So what amount do we bitshift by? If we just want to get the original bitstring, we would shift by `(num_qubits - 1 - arange(num_qubits))`. For a four qubit system, this gives `[3, 2, 1, 0]`. Looking at our previous example, `[4] >> [3, 2, 1, 0] gives [0, 1, 2, 4]` and `&`ing with `1` gives `[0, 1, 0, 0]` which we can read as `0100` or 4, in a four digit binary representation.

Now, we don't just want the bit representations&mdash;we want to permute them. So if we permute the order we do the bitshifts in, we will end up with permuted bitstrings. If we have a three qubit system and want to place the rightmost qubit first, then the first bitshift we want to do is zero, or no shift. That's `num_qubits - 1 - 2`. If we want to put the leftmost qubit in the middle, we want to do two bitshifts next: `num_qubits - 1 - 0`. Finally, if we want the middle qubit last, we want to do one bitshift, which is `num_qubits - 1 - 1`. That is, we want the list we give to our bitshift calculation to be `[2, 0, 1]`. This is exactly the permuted list of qubit indices calculated in `permuted_qubits`.

To reinterpret these lists of bits (in `bits`) as state vector indices, we do the bit manipulations in `permuted_state_indices`. Each list of bits needs to somehow be combined back into one bitstring. We can do this by shifting each bit in the list back to the left the correct number of places and then summing each list of bits (separately from each other, of course). The result of the sum will be the correct bitstring, which will of course give the correct index.

The amount by which to bitshift is dictated by the order in the list. If we have a three qubit system, the first element in the list of qubits needs to end up in the fours place, which means it needs to be shifted twice. That's `num_qubits - 1 - 0`. The second element needs to end up in the twos place, which means a shift of one&mdash;`num_qubits - 1 - 1`. And the last element needs to be in the ones place, which is of course no shift at all, or `num_qubits - 1 - 2`. That is, the list we give to the bitshift calculation is `[0, 1, 2]`; hence we do `bits << (num_qubits - 1 - arange(num_qubits))`. For the list of bits giving `[4]`, we would have (after bitshifts) `[0000, 100, 00, 0]`. Summing gives `0100`, or 4. So after summing along the horizontal axis, we have a permuted list of state vector indices.

### construct_operator

Turns a `GateApplication` on $m$ qubits into an `Operator` on $n$ qubits. Takes the `GateApplication` to convert, and a list of qubits permuted into the order that they should be in for the `Operator` to actually have the expected behavior. That is, if we have a 4 qubit circuit and a CNOT gate on qubits (2,0), then `permuted_qubits` is something like `[2,0,1,3]`.

### invert_permutation

Takes a `permutation_vector` and constructs the inverse.

### \_add\_gate

Constructs a `GateApplication` and adds it to the circuit's list of gates. Accepts an `Operator` and an arbitrary number of `qubits`. Sets `compiled` to false. Returns the circuit. Mostly intended as an internal helper function.

### h

Constructs a Hadamard `Operator` acting on the specified `qubit` and appends it to `gates`.

### x

Constructs a Pauli X `Operator` acting on the specified `qubit` and appends it to `gates`.

### y

Constructs a Pauli Y `Operator` acting on the specified `qubit` and appends it to `gates`.

### z

Constructs a Pauli Z `Operator` acting on the specified `qubit` and appends it to `gates`.

### t

Constructs a T `Operator` acting on the specified `qubit` and appends it to `gates`.

### phase

Constructs a phase `Operator` acting on the specified `qubit` and appends it to `gates`.

### identity

Constructs an identity `Operator` acting on the specified `qubit` and appends it to `gates`.

### rx

Constructs an X rotation `Operator` from angle `theta` acting on the specified `qubit` and appends it to `gates`.

### ry

Constructs a Y rotation `Operator` from angle `theta` acting on the specified `qubit` and appends it to `gates`.

### rz

Constructs a Z rotation `Operator` from angle `theta` acting on the specified `qubit` and appends it to `gates`.

### phase_shift

Constructs a phase shift `Operator` from angle `phi` acting on the specified `qubit` and appends it to `gates`.

### u

Constructs a U `Operator` from angles `theta`, `phi`, and `lam`bda acting on the specified `qubit` and appends it to `gates`.

### cnot

Constructs a CNOT `Operator` acting on the specified `control` and `target` and appends it to `gates`.

### cz

Constructs a CZ `Operator` acting on the specified `control` and `target` and appends it to `gates`.

### swap

Constructs a swap `Operator` acting on the specified `targets` and appends it to `gates`.

### toffoli

Constructs a Toffoli `Operator` acting on the specified `controls` and `target` and appends it to `gates`.

### fredkin

Constructs a Fredkin `Operator` acting on the specified `control` and `targets` and appends it to `gates`.

### custom_gate

Constructs a `GateApplication` from `Operator` applied to `qubits`. If the qubit ordering is important, that ordering should be reflected in the input provided.

### \_\_str\_\_

Constructs a CLI-friendly circuit diagram. Represents each gate using its `Operator`'s `symbol`. 

### \_print\_multi\_gate

Helper function for `__str__`. Correctly constructs the diagram representations of multi-qubit gates, including which are controls vs. targets.

## GateApplication

This one is pretty simple&mdash;just a dataclass that stores an `Operator` and a tuple of `qubits` (indices). There is one member function though.

### \_\_post\_init\_\_

Verifies that the provided operator and qubit indices make sense together&mdash;checks that there aren't more qubits provided than the `Operator` can act on, and checks that there are no duplicate qubit indices.