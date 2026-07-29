from operators import Operator
from gate_application import GateApplication
from state_vector import StateVector
from numpy import eye, array, ndarray, arange, str_
from numpy.typing import NDArray
from typing import Iterable

WIRE = "\u2500"
VERTICAL = "\u2502"
JUNCTION = "\u253C"
CONTROL = "\u25A0"
MULT = "\u00D7"

class Circuit():

    def __init__(self, num_qubits: int, gates: list[GateApplication]=None):
        
        self.gates = gates if gates else list()
        self.num_qubits = num_qubits
        self.instructions: list[Operator | list[int]] = []
        self.compiled = False


    def run(self, initial_state: StateVector=None) -> StateVector:
        """
        Compile the circuit and run the simulation.
        """
        if not self.compiled:
            self.compile()

        if not initial_state:
            initial_state = StateVector.from_index(0, self.num_qubits)

        return self.simulate(initial_state)


    def sample(self, num_samples: int, initial_state: StateVector=None, seed: int=None) -> dict[str_, int]:
        """
        Run the provided state through the circuit and take num_samples measurements of it.
        """
        state = self.run(initial_state)

        return state.sample(num_samples, seed)

    
    def simulate(self, initial_state: StateVector) -> StateVector:
        """
        Simulate the action of the compiled circuit on a StateVector.
        """
        if initial_state.num_qubits != self.num_qubits:
            raise ValueError(f"Circuit expects a {self.num_qubits} qubit state vector, got {initial_state.num_qubits}")

        if not self.compiled:
            raise ValueError("Circuit must be compiled before simulating.")

        state = initial_state.copy()
        
        for step in self.instructions:
            # Steps in the circuit are either Operator objects or permutation vectors. Pick the right action.
            if isinstance(step, ndarray) and step.ndim == 1:
                state.permute_state(step)

            elif isinstance(step, Operator):
                state.evolve(step)

            else:
                raise ValueError(f"Invalid instruction type in circuit: {type(step)}")

        return state

    
    def compile(self):
        """
        Turn the circuit's list of GateApplications into a list of operators that can actually be applied to
        a StateVector.
        """
        self.instructions.clear()

        for gate in self.gates:
            qubits = gate.qubits

            if any(q >= self.num_qubits or q < 0 for q in qubits):
                raise ValueError(f"Qubit index out of range. Maximum index: {self.num_qubits - 1}")

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
                self.instructions.append(permuted_state_indices)
                self.instructions.append(self.construct_operator(gate, permuted_qubits))
                self.instructions.append(self.invert_permutation(permuted_state_indices))

            # If the operator only acts on one qubit at a time, just add it to the list.
            else:
                self.instructions.append(self.construct_operator(gate, range(self.num_qubits)))

        self.compiled = True

    
    def construct_operator(self, gate: GateApplication, ordered_qubits: Iterable[list[int]]) -> Operator:
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


    def invert_permutation(self, permutation_vector: list[int]) -> ndarray[int]:
        """
        Compute a permutation vector to invert a given permutation.
        """
        inverse = [0] * len(permutation_vector)

        for i, j in enumerate(permutation_vector):
            inverse[j] = i

        return array(inverse)
    

    def _add_gate(self, operator: Operator, *qubits: int):
        """
        Add a GateApplication to the circuit's collection of gates.

        Args:
            operator: the Operator object to add.
            qubits: the qubits the gate will act on.
        """
        self.gates.append(GateApplication(operator, qubits))
        self.compiled = False
        return self


    # The following are all functionally identical, so they don't get individual comments. They're just dedicated
    # constructors for each type of gate. Hopefully self explanatory.


    def h(self, qubit: int):
        return self._add_gate(Operator.hadamard(), qubit)


    def x(self, qubit: int):
        return self._add_gate(Operator.pauli_x(), qubit)


    def y(self, qubit: int):
        return self._add_gate(Operator.pauli_y(), qubit)


    def z(self, qubit: int):
        return self._add_gate(Operator.pauli_z(), qubit)


    def t(self, qubit: int):
        return self._add_gate(Operator.t(), qubit)


    def phase(self, qubit: int):
        return self._add_gate(Operator.phase(), qubit)


    def identity(self, qubit: int):
        return self._add_gate(Operator.identity(1), qubit)


    def rx(self, qubit: int, theta: float):
        return self._add_gate(Operator.rx(theta), qubit)


    def ry(self, qubit: int, theta: float):
        return self._add_gate(Operator.ry(theta), qubit)


    def rz(self, qubit: int, theta: float):
        return self._add_gate(Operator.rz(theta), qubit)


    def phase_shift(self, qubit: int, phi: float):
        return self._add_gate(Operator.phase_shift(phi), qubit)


    def u(self, qubit: int, theta: float, phi: float, lam: float):
        return self._add_gate(Operator.u(theta, phi, lam), qubit)


    def cnot(self, control: int, target: int):
        if control == target:
            raise ValueError("Control qubit equals target qubit")

        return self._add_gate(Operator.cnot(), control, target)


    def cz(self, control: int, target: int):
        if control == target:
            raise ValueError("Control qubit equals target qubit")
        
        return self._add_gate(Operator.cz(), control, target)


    def swap(self, target_1: int, target_2: int):
        if target_1 == target_2:
            raise ValueError("Cannot swap a qubit with itself")
        
        return self._add_gate(Operator.swap(), target_1, target_2)


    def toffoli(self, control_1: int, control_2: int, target: int):
        if len({control_1, control_2, target}) < 3:
            raise ValueError("Control or target qubit equals another control or target qubit. ",
                             f"control_1: {control_1}, control_2: {control_2}, target: {target}")

        return self._add_gate(Operator.toffoli(), control_1, control_2, target)


    def fredkin(self, control: int, target_1: int, target_2: int):
        if len({control, target_1, target_2}) < 3:
            raise ValueError("Control or target qubit equals another control or target qubit. ",
                                f"control: {control}, target_1: {target_1}, target_2: {target_2}")

        return self._add_gate(Operator.fredkin(), control, target_1, target_2)


    def custom_gate(self, operator: Operator, *qubits: int):
        if len(set(qubits)) < len(qubits):
            raise ValueError("Gate cannot be applied multiple times to one qubit")

        return self._add_gate(operator, qubits)


    def __str__(self) -> str:
        """
        Constructs a text-based circuit diagram.
        """
        if not self.gates:
            print(f"{self.num_qubits} qubit circuit with no gates added.")

        # Initialize the data structure we'll use to construct the output.
        # This is pretty straightforward: a list of strings, where each string in the list corresponds to
        # one line of output. Qubit lines start with a qubit, off lines just get spaces.
        lines = [start for i in range(self.num_qubits) for start in (f"q{i} " + 3 * WIRE, "      ")]
        lines.pop()

        # Work our way through the gates and add them to the diagram.
        for gate in self.gates:
            qubits = gate.qubits
            symbol = gate.operator.symbol

            # Relatively easy to handle if it's a one qubit gate:
            if len(qubits) == 1:
                for i in range(self.num_qubits):
                    # Go through the qubits in the sim and when you find the right one, add the gate to that line
                    if i in qubits:
                        lines[2 * i] += (symbol + 3 * WIRE)

                    # Otherwise, add more circuit lines and whitespace
                    else:
                        lines[2 * i] += (3 + len(symbol)) * WIRE

                    if i < self.num_qubits - 1:
                        lines[2 * i + 1] += (3 + len(symbol)) * " "

            # If the gate is multi-qubit, we use a helper function to avoid excessive code duplication.
            else:
                match gate.operator.symbol.lower():
                    case "cnot":
                        self._print_multi_gate(lines, gate, "X", 1)
                        
                    case "cz":
                        self._print_multi_gate(lines, gate, "Z", 1)
                        
                    case "swap":
                        self._print_multi_gate(lines, gate, MULT, 0)

                    case "toff":
                        self._print_multi_gate(lines, gate, "X", 2)
                        
                    case "fred":
                        self._print_multi_gate(lines, gate, MULT, 1)
                        
                    case _:
                        raise ValueError(f"Unrecognized operator symbol: {gate.operator.symbol}")

        output = str()

        # Add newlines and concatenate all the lines of output together
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                line += "\n"
            output += line

        return output        


    def _print_multi_gate(self, lines: list[str], gate: GateApplication, target_symbol: str, num_controls: int):
        """
        Helper method to construct more complicated gate diagrams.
        """
        qubits = gate.qubits

        # I'm establishing a convention that the first n qubits listed in a GateApplication's qubits are the controls.
        control_bits = [qubit for i, qubit in enumerate(qubits) if i < num_controls]
        target_bits = set(qubits) - set(control_bits)

        # This is the range of qubits that are actually affected by the gate (diagrammatically, anyway).
        qubit_indices = arange(min(qubits), max(qubits) + 1)

        for i in range(self.num_qubits):

            # When we find a qubit in the right range, either add the control symbol,
            if i in qubit_indices:
                if i in control_bits:
                    lines[2 * i] += (CONTROL + (2 + len(target_symbol)) * WIRE)

                # target symbol,
                elif i in target_bits:
                    lines[2 * i] += (target_symbol + 3 * WIRE)

                # or junction symbol, depending on if/how the qubit is involved in the actual action of the gate.
                else:
                    lines[2 * i] += (JUNCTION + (2 + len(target_symbol)) * WIRE)

                # Also add whitespace and a vertical bar to the off lines.
                if i < max(qubits):
                    lines[2 * i + 1] += (VERTICAL + (2 + len(target_symbol)) * " ")

            # If we're not in the right range, just add circuit lines or whitespace as appropriate
            else:
                lines[2 * i] += (3 + len(target_symbol)) * WIRE
                
                if i < self.num_qubits - 1:
                    lines[2 * i + 1] += (3 + len(target_symbol)) * " "
