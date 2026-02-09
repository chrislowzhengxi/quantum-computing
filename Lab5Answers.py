# %% [markdown]
# # Lab 5: Simulating an N-qubit system
# 
# ## Introduction: Building on Single Qubit Systems
# 
# Last week, you built a `SingleQubitSystem` class that simulated quantum states and operations on a single qubit. You implemented core methods to initialize quantum states, inspect them (measuring probabilities), and apply quantum gates like Hadamard and Pauli-Z. This week, we're taking a significant step forward: you'll generalize that single-qubit simulator to handle an **arbitrary number of qubits**.
# 
# In the previous assignment, you were provided with substantial skeleton code that outlined the class structure, method signatures, and helper functions. This scaffolding allowed you to focus on understanding the quantum mechanics and implementing the core logic. **In this assignment, you will be expected to do everything yourself.** You'll implement all required methods, and write comprehensive tests.
# 
# 
# The jump from single-qubit to N-qubit systems requires a fundamental shift in how you think about quantum state representation.
# 
# In the `SingleQubitSystem`, you could hardcode quantum gate matrices—a 2×2 matrix for single-qubit gates is simple and manageable. Now, with N qubits, gate matrices scale exponentially: an operation on an N-qubit system requires a 2^N × 2^N matrix. Rather than hardcoding these, you'll need to **programmatically construct** gate matrices and tensor products that generalize to any number of qubits.
# 
# Beyond extending existing operations, in this lab you will also implement:
# 
# - **Multi-qubit gates**: You'll implement the SWAP and the CNOT gates.
# - **Quantum oracles**: You'll implement two oracles—the **BernVaz oracle** and the **Archimedes oracle**.
# 
# # Task 0: Swapping bits
# 
# Before we dive into quantum mechanics, let's build intuition for how **SWAP** operations work at the bit level. Understanding classical bit swapping will help you reason about the quantum SWAP gate you'll implement later.
# 
# We've provided you with a helper function that converts integers to binary strings in **big-endian**:
# 

# %%
def int_to_bin_string(n: int, number_of_bits: int) -> str:
    """
    Convert an integer to a binary string in big-endian format.

    Args:
        n: The integer to convert
        number_of_bits: The length of the output binary string

    Returns:
        A binary string of length number_of_bits in big-endian format

    """
    return format(n, f'0{number_of_bits}b')


# %% [markdown]
# 
# A 4-bit system (called a **nibble**) can represent 16 different states (0 through 15). Your task is to print a truth table showing what happens when you swap the bits at **index 0** and **index 2**.
# 
# The truth table should have three columns:
# 
# 1. **Original**: The original 4-bit string
# 2. **After Swap**: The 4-bit string after swapping bits at positions 0 and 2
# 3. **Decimal**: The decimal value of the resulting bit string

# %%
# Format your table like this:
original = after_swap = int_to_bin_string(3,4)
print(f'{original}\t{after_swap}')

## TODO: Print the truth table after swapping bits at positions 0 and 2

def swap_bits(bit_string: str, i: int, j: int) -> str:
    bits = list(bit_string)
    bits[i], bits[j] = bits[j], bits[i]
    return "".join(bits)


print("Original\tAfter Swap\tDecimal")

for n in range(16):
    original = int_to_bin_string(n, 4)
    swapped = swap_bits(original, 0, 2)
    decimal = int(swapped, 2)

    print(f"{original}\t{swapped}\t{decimal}")



# %% [markdown]
# 
# Now that you've seen how bit swapping works at the classical level, think about how this concept might extend to quantum systems.
# 
# Consider these questions as you reflect:
# 
# 1. **State representation**: In your `SingleQubitSystem`, you represented a quantum state as a state vector—a list of amplitudes, one for each possible classical state. How might you use bit swapping to rearrange the elements of an N-qubit state vector?
# 2. **Permutation and basis states**: When you swap bits in a classical state, you're essentially permuting which basis state corresponds to which computational outcome. If bit swapping changes the binary representation, how does that change *which* element of the state vector you're modifying?
# 3. **Generalizing to multiple qubits**: In a 2-qubit system, there are 4 basis states ($|00\rangle$, $|01\rangle$, $|10\rangle$, $|11\rangle$). If you wanted to swap qubits 0 and 1, how would the indices of your state vector elements need to be rearranged? Can you see a pattern that would work for *any* pair of qubits in an N-qubit system?
# 4. **Efficiency**: Rather than constructing a full $2^N \times 2^N$ matrix, could you implement SWAP by directly rearranging elements of a state vector? What would be the computational advantage?
# 

# %% [markdown]
# 
# ## `NQubitSystem` technical specs
# 
# `NQubitSystem` is a concrete implementation that inherits from the abstract `QubitSystem` base class. It generalizes single-qubit operations to support quantum systems with an arbitrary number of qubits.
# 
# 
# | Function | Behavior |
# | :-- | :-- |
# | `__init__(num_qubits: int)` | Initializes an N-qubit system with `num_qubits` qubits. The initial state should be the computational basis state $\lvert 0 \ldots 0 \rangle$ (all qubits in state $\lvert 0 \rangle$). Raise a `ValueError` if `num_qubits` is less than 1. |
# | `set_value(value: list[float])` | Sets the system's quantum state. Accept a list of floats representing the state vector amplitudes in **big-endian** basis order. The length of the input must equal $2^{\text{num_qubits}}$. Implementations should validate that the input represents a valid state (i.e., the sum of squared amplitudes equals 1.0, within numerical precision) and raise a `ValueError` if not. |
# | `get_value_braket() -> str` | Returns a bra-ket style string representing the current state. This is primarily for readability/debugging, you won't be penalized for stylistic differences or formatting. |
# | `get_value_vector() -> list[float]` | Returns the current state vector amplitudes in big-endian basis order. |
# | `apply_not(i: int) -> None` | Applies the NOT gate (Pauli-$X$) to qubit at index $i$, updating the system state. Raise an `IndexError` if $i$ is not a valid qubit index. |
# | `apply_h(i: int) -> None` | Applies the Hadamard gate ($H$) to qubit at index $i$, updating the system state. Raise an `IndexError` if $i$ is not a valid qubit index. |
# | `apply_z(i: int) -> None` | Applies the Pauli-$Z$ gate to qubit at index $i$, updating the system state. Raise an `IndexError` if $i$ is not a valid qubit index. |
# | `apply_cnot(control: int, target: int) -> None` | Applies a controlled-NOT gate with `control` as the control qubit and `target` as the target qubit, updating the system state. Raise an `IndexError` if either index is invalid, or if they are the same. |
# | `apply_swap(i: int, j: int) -> None` | Swaps qubits at indices $i$ and $j$, updating the system state. Raise an `IndexError` if either index is invalid, or if they are the same. |
# | `measure() -> str` | <p>Simulates a measurement of the state of the system and returns one of the possible values as a big-endian string of binary. For example, if the state is $\lvert 000 \rangle$, the result would always be `'000'`. If the state is $\frac{1}{\sqrt{2}} \lvert 000 \rangle + \frac{1}{\sqrt{2}} \lvert 101 \rangle$, measurement would return `'000'` or `'101'` with equal probability (50% each).<ul><li>Note: If a system is in a state of superposition before `measure()`, the act of measurement should collapse the superposition.</li><li>Note 2: The output should always have the same number of bits as the system does. For a 3-qubit system, outputs will be 3-character strings like `'000'`, `'101'`, etc.</li></ul></p> |
# 
# 
# ## Using Numpy
# 
# 
# As with last week's lab, you are free to use `numpy` to handle mathematical operations. You may **not** use `qiskit` or **any** other library besides `numpy` as part of your solution. **Adding extra `import` statements _will_ crash the autograder.**
# 
# You can perform vector-matrix multiplication with `numpy` as follows:

# %%
import numpy as np

# Define a state vector (e.g., |01⟩ for a 2-qubit system)
state = np.array([0, 1, 0, 0])

# Define a gate (e.g., a 2-qubit gate, X on the second qubit)
gate = np.array([[0, 1, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

# Apply the gate to the state using matrix-vector multiplication (Note: Numpy treats vectors as row-vectors)
new_state = gate @ state
print(new_state)
# Output: [1 0 0 0]

# Alternatively, using np.dot()
new_state = np.dot(gate, state)
print(new_state)

# %% [markdown]
# To compute the tensor product (also called the Kronecker product):

# %%
import numpy as np

# Define two small matrices (e.g., single-qubit gates)
I = np.array([[1, 0],
              [0, 1]])

X = np.array([[0, 1],
              [1, 0]])

# Tensor product
result = np.kron(I, X)
print(result)

# Output:
#[[0 1 0 0]
# [1 0 0 0]
# [0 0 0 1]
# [0 0 1 0]]


# %% [markdown]
# ## Task 1: Copy over your `QubitSystem` implementation from last week's assignment

# %%
## TODO: Paste QubitSystem here
from abc import ABC, abstractmethod
import random
import math
import numpy as np

class QubitSystem(ABC):
    """
    Abstract interface for a system of qubits.

    Child classes should store and update the quantum state, and must raise
    IndexError when gate indices are out of bounds.
    """

    def __init__(self, num_qubits: int):
        self._num_qubits = num_qubits
        self.state = [0]*2**num_qubits
        self.state[0] = 1

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def _check_index(self, i: int) -> None:
        """Helper for consistent index-out-of-bounds behavior."""
        if i < 0 or i >= self._num_qubits:
            raise IndexError(f"Qubit index out of bounds: {i}")

    def _check_pair(self, i: int, j: int) -> None:
        """Helper for two-qubit operations."""
        self._check_index(i)
        self._check_index(j)

    @abstractmethod
    def set_value(self, value) -> None:
        """Set the current state using bra-ket string or state-vector list."""
        raise NotImplementedError

    @abstractmethod
    def get_value_braket(self) -> str:
        """Return a bra-ket formatted string for the current state."""
        raise NotImplementedError

    @abstractmethod
    def get_value_vector(self) -> list[float]:
        """Return the state vector (big-endian basis ordering)."""
        raise NotImplementedError

    @abstractmethod
    def apply_not(self, i: int) -> None:
        """Apply NOT (Pauli-X) to qubit i."""
        raise NotImplementedError

    @abstractmethod
    def apply_h(self, i: int) -> None:
        """Apply Hadamard to qubit i."""
        raise NotImplementedError

    @abstractmethod
    def apply_z(self, i: int) -> None:
        """Apply Pauli-Z to qubit i."""
        raise NotImplementedError

    @abstractmethod
    def apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT with given control and target qubits."""
        raise NotImplementedError

    @abstractmethod
    def apply_swap(self, i: int, j: int) -> None:
        """Swap qubits i and j."""
        raise NotImplementedError

    @abstractmethod
    def measure(self) -> str:
        """Simualte a measurement of the system."""
        raise NotImplementedError

class SingleQubitSystem(QubitSystem):
    def __init__(self):
        super().__init__(num_qubits=1)

    def set_value(self, value) -> None:
        """Set qubit state from a list or bra-ket string."""
        if isinstance(value, str):
            if value == "|0>":
                self.state = [1.0, 0.0]
            elif value == "|1>":
                self.state = [0.0, 1.0]
            elif value == "|+>":
                # |+> = (1/sqrt(2))|0> + (1/sqrt(2))|1>
                self.state = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
            elif value == "|->":
                # |-> = (1/sqrt(2))|0> - (1/sqrt(2))|1>
                self.state = [1.0 / math.sqrt(2), -1.0 / math.sqrt(2)]
            else:
                raise ValueError(f"Invalid bra-ket string: {value}")
        elif isinstance(value, list):
            if len(value) != 2:
                raise ValueError(f"Single qubit state must have 2 amplitudes, got {len(value)}")
            
            norm_sq = sum(abs(amp)**2 for amp in value)
            if not (0.99 <= norm_sq <= 1.01): 
                raise ValueError(f"State vector not normalized: sum of |amplitude|^2 = {norm_sq}")
            
            self.state = [float(amp) for amp in value]
        else:
            raise ValueError(f"Expected list or string, got {type(value)}")

    def get_value_braket(self) -> str:
        """Return bra-ket representation of the state."""
        alpha, beta = self.state[0], self.state[1]
        
        terms = []
        
        if abs(alpha) > 1e-10:
            if abs(abs(alpha) - 1.0) < 1e-10:
                terms.append("|0>")
            else:
                terms.append(f"{alpha:.4f}|0>")
        
        if abs(beta) > 1e-10:
            sign = "+" if beta > 0 else "-"
            abs_beta = abs(beta)
            if abs(abs_beta - 1.0) < 1e-10:
                if terms:
                    terms.append(f"{sign} |1>")
                else:
                    terms.append("|1>" if beta > 0 else "-|1>")
            else:
                if terms:
                    terms.append(f"{sign} {abs_beta:.4f}|1>")
                else:
                    terms.append(f"{beta:.4f}|1>")
        
        if not terms:
            return "0"
        
        return " ".join(terms)

    def get_value_vector(self) -> list[float]:
        """Return the state vector."""
        return self.state.copy()

    def apply_not(self, i: int) -> None:
        """Apply Pauli-X (NOT) gate to qubit i."""
        self._check_index(i)
        # NOT gate: |0> -> |1>, |1> -> |0>
        # Matrix: [[0, 1], [1, 0]]
        alpha, beta = self.state[0], self.state[1]
        self.state = [beta, alpha]

    def apply_h(self, i: int) -> None:
        """Apply Hadamard gate to qubit i."""
        self._check_index(i)
        # H = (1/sqrt(2)) * [[1, 1], [1, -1]]
        alpha, beta = self.state[0], self.state[1]
        inv_sqrt2 = 1.0 / math.sqrt(2)
        self.state = [
            inv_sqrt2 * (alpha + beta),
            inv_sqrt2 * (alpha - beta)
        ]

    def apply_z(self, i: int) -> None:
        """Apply Pauli-Z gate to qubit i."""
        self._check_index(i)
        # Z gate: Matrix [[1, 0], [0, -1]]
        alpha, beta = self.state[0], self.state[1]
        self.state = [alpha, -beta]

    def apply_cnot(self, control: int, target: int) -> None:
        """CNOT is not defined for single qubit system."""
        raise IndexError("Cannot apply CNOT to single-qubit system")

    def apply_swap(self, i: int, j: int) -> None:
        """SWAP is not defined for single qubit system."""
        raise IndexError("Cannot apply SWAP to single-qubit system")

    def measure(self) -> str:
        """Measure the qubit, collapsing the state."""
        alpha, beta = self.state[0], self.state[1]
        
        prob_0 = abs(alpha) ** 2
        
        if random.random() < prob_0:
            # Measure |0>
            self.state = [1.0, 0.0]
            return "0"
        else:
            # Measure |1>
            self.state = [0.0, 1.0]
            return "1"


# %% [markdown]
# ## Task 2: Adapt your test suite from `SingleQubitSystem` for `NQubitSystem`
# Just like last week, the autograder will expect the following tests to be present in your submission: `test_set_value`,`test_get_value_vector`,`test_apply_not`,`test_apply_h`,`test_apply_z`,`test_apply_cnot`,`test_apply_swap`,`test_measure`

# %%
## TODO: Implement your test suite here
from abc import ABC, abstractmethod
import random
import math
import numpy as np

def compare_lists(first_list, second_list, eps: float = 1e-3) -> bool:
    import numpy as np
    return bool(np.allclose(first_list, second_list, atol=eps, rtol=0.0))


def test_set_value() -> bool:
    # Arrange
    q2 = NQubitSystem(2)

    # Test 1: Valid 2-qubit state
    q2.set_value([1.0, 0.0, 0.0, 0.0])
    if not compare_lists(q2.state, [1.0, 0.0, 0.0, 0.0]):
        return False

    # Test 2: Valid superposition
    inv_sqrt2 = 1.0 / (2**0.5)
    q2.set_value([inv_sqrt2, 0.0, inv_sqrt2, 0.0])
    if not compare_lists(q2.state, [inv_sqrt2, 0.0, inv_sqrt2, 0.0]):
        return False

    # Test 3: Invalid length
    try:
        q2.set_value([1.0, 0.0, 0.0])
        return False
    except ValueError:
        pass

    # Test 4: Invalid normalization
    try:
        q2.set_value([2.0, 0.0, 0.0, 0.0])
        return False
    except ValueError:
        pass

    # Test 5: Invalid type
    try:
        q2.set_value("|00>")
        return False
    except ValueError:
        pass

    # Test 6: Valid 3-qubit state length
    q3 = NQubitSystem(3)
    q3.set_value([1.0] + [0.0]*7)
    if not compare_lists(q3.state, [1.0] + [0.0]*7):
        return False

    
    # Extra: Valid 1-qubit state length
    q1 = NQubitSystem(1)
    q1.set_value([0.0, 1.0])  # |1>
    if not compare_lists(q1.state, [0.0, 1.0]):
        return False

    # Extra: Valid 4-qubit state length
    q4 = NQubitSystem(4)
    q4.set_value([1.0] + [0.0]*15)  # |0000>
    if not compare_lists(q4.state, [1.0] + [0.0]*15):
        return False
    
    return True


def test_set_value_rejects_non_list() -> bool:
    q = NQubitSystem(2)
    try:
        q.set_value("not a list")
        return False
    except ValueError:
        return True


def test_init_invalid_num_qubits() -> bool:
    try:
        _ = NQubitSystem(0)
        return False
    except ValueError:
        return True


def test_get_value_vector() -> bool:
    q = NQubitSystem(3)
    state = [0.0]*8
    state[5] = 1.0  # |101>
    q.set_value(state)

    vec = q.get_value_vector()
    if not compare_lists(vec, state):
        return False

    # Ensure copy
    vec[5] = 0.0
    if not compare_lists(q.state, state):
        return False

    return True


def test_apply_not() -> bool:
    q = NQubitSystem(2)

    # Test 1: NOT on qubit 0 maps |00> -> |10>
    q.set_value([1.0, 0.0, 0.0, 0.0])
    q.apply_not(0)
    if not compare_lists(q.state, [0.0, 0.0, 1.0, 0.0]):
        return False

    # Test 2: NOT on qubit 1 maps |00> -> |01>
    q.set_value([1.0, 0.0, 0.0, 0.0])
    q.apply_not(1)
    if not compare_lists(q.state, [0.0, 1.0, 0.0, 0.0]):
        return False

    # Test 3: Invalid index
    try:
        q.apply_not(2)
        return False
    except IndexError:
        pass

    try:
        q.apply_not(-1)
        return False
    except IndexError:
        pass

    return True


def test_apply_not_three_qubits_mapping() -> bool:
    q = NQubitSystem(3)
    q.set_value([1.0] + [0.0] * 7)  # |000>
    q.apply_not(1)  # flip middle qubit
    expected = [0.0] * 8
    expected[2] = 1.0  # |010>
    return compare_lists(q.state, expected)


def test_apply_not_big_endian_mapping() -> bool:
    q = NQubitSystem(3)
    state = [0.0] * 8
    state[1] = 1.0  # |001>
    q.set_value(state)
    q.apply_not(0)  # flip MSB
    expected = [0.0] * 8
    expected[5] = 1.0  # |101>
    return compare_lists(q.state, expected)


def test_apply_h() -> bool:
    q = NQubitSystem(2)
    inv_sqrt2 = 1.0 / (2**0.5)

    # Test 1: H on qubit 0: |00> -> (|00> + |10>)/sqrt2
    q.set_value([1.0, 0.0, 0.0, 0.0])
    q.apply_h(0)
    if not compare_lists(q.state, [inv_sqrt2, 0.0, inv_sqrt2, 0.0]):
        return False

    # Test 2: H on qubit 1: |00> -> (|00> + |01>)/sqrt2
    q.set_value([1.0, 0.0, 0.0, 0.0])
    q.apply_h(1)
    if not compare_lists(q.state, [inv_sqrt2, inv_sqrt2, 0.0, 0.0]):
        return False

    # Test 3: Invalid index
    try:
        q.apply_h(2)
        return False
    except IndexError:
        pass
    
    # Test 4: H on |10> gives (|00> - |10>)/sqrt2
    q.set_value([0.0, 0.0, 1.0, 0.0])  # |10>
    q.apply_h(0)
    if not compare_lists(
        q.state,
        [inv_sqrt2, 0.0, -inv_sqrt2, 0.0]
    ):
        return False
    
    q1 = NQubitSystem(1)
    q1.set_value([0.0, 1.0])  # |1>
    q1.apply_h(0)
    if not compare_lists(q1.state, [inv_sqrt2, -inv_sqrt2]):
        return False


    return True


def test_apply_h_then_h_identity() -> bool:
    q = NQubitSystem(2)
    state = [0.0, 1.0, 0.0, 0.0]  # |01>
    q.set_value(state)
    q.apply_h(0)
    q.apply_h(0)
    return compare_lists(q.state, state)


def test_apply_z() -> bool:
    q = NQubitSystem(2)

    # Test 1: Z on qubit 0 flips phase of |10>
    q.set_value([0.0, 0.0, 1.0, 0.0])
    q.apply_z(0)
    if not compare_lists(q.state, [0.0, 0.0, -1.0, 0.0]):
        return False

    # Test 2: Z on qubit 1 flips phase of |01>
    q.set_value([0.0, 1.0, 0.0, 0.0])
    q.apply_z(1)
    if not compare_lists(q.state, [0.0, -1.0, 0.0, 0.0]):
        return False

    # Test 3: Invalid index
    try:
        q.apply_z(2)
        return False
    except IndexError:
        pass

    # Test 4: Z on |00> does nothing
    q.set_value([1.0, 0.0, 0.0, 0.0])
    q.apply_z(0)
    if not compare_lists(q.state, [1.0, 0.0, 0.0, 0.0]):
        return False


    return True


def test_apply_z_hzh_equals_x() -> bool:
    q = NQubitSystem(2)
    q.set_value([1.0, 0.0, 0.0, 0.0])  # |00>
    q.apply_h(1)
    q.apply_z(1)
    q.apply_h(1)
    expected = [0.0, 1.0, 0.0, 0.0]  # |01>
    return compare_lists(q.state, expected)


def test_apply_cnot() -> bool:
    q = NQubitSystem(2)

    # Test 1: control 0 target 1, |10> -> |11>
    q.set_value([0.0, 0.0, 1.0, 0.0])
    q.apply_cnot(0, 1)
    if not compare_lists(q.state, [0.0, 0.0, 0.0, 1.0]):
        return False

    # Test 2: control 0 target 1, |00> unchanged
    q.set_value([1.0, 0.0, 0.0, 0.0])
    q.apply_cnot(0, 1)
    if not compare_lists(q.state, [1.0, 0.0, 0.0, 0.0]):
        return False

    # Test 3: control 1 target 0, |01> -> |11>
    q.set_value([0.0, 1.0, 0.0, 0.0])
    q.apply_cnot(1, 0)
    if not compare_lists(q.state, [0.0, 0.0, 0.0, 1.0]):
        return False

    # Test 4: Invalid indices
    try:
        q.apply_cnot(0, 0)
        return False
    except IndexError:
        pass

    try:
        q.apply_cnot(2, 1)
        return False
    except IndexError:
        pass

    return True


def test_apply_cnot_control_off() -> bool:
    q = NQubitSystem(2)
    q.set_value([0.0, 1.0, 0.0, 0.0])  # |01> control=0 off
    q.apply_cnot(0, 1)
    return compare_lists(q.state, [0.0, 1.0, 0.0, 0.0])


def test_apply_cnot_negative_index() -> bool:
    q = NQubitSystem(2)
    try:
        q.apply_cnot(-1, 1)
        return False
    except IndexError:
        return True


def test_apply_cnot_three_qubits() -> bool:
    q = NQubitSystem(3)
    state = [0.0] * 8
    state[4] = 1.0  # |100>
    q.set_value(state)
    q.apply_cnot(0, 2)
    expected = [0.0] * 8
    expected[5] = 1.0  # |101>
    return compare_lists(q.state, expected)


def test_apply_cnot_creates_bell_state() -> bool:
    q = NQubitSystem(2)
    inv_sqrt2 = 1.0 / (2**0.5)
    q.set_value([1.0, 0.0, 0.0, 0.0])
    q.apply_h(0)
    q.apply_cnot(0, 1)
    expected = [inv_sqrt2, 0.0, 0.0, inv_sqrt2]
    return compare_lists(q.state, expected)


def test_apply_swap() -> bool:
    q = NQubitSystem(2)

    # Test 1: swap(0,1) on |01> -> |10>
    q.set_value([0.0, 1.0, 0.0, 0.0])
    q.apply_swap(0, 1)
    if not compare_lists(q.state, [0.0, 0.0, 1.0, 0.0]):
        return False

    # Test 2: swap(0,1) on |10> -> |01>
    q.set_value([0.0, 0.0, 1.0, 0.0])
    q.apply_swap(0, 1)
    if not compare_lists(q.state, [0.0, 1.0, 0.0, 0.0]):
        return False

    # Test 3: Invalid indices
    try:
        q.apply_swap(0, 0)
        return False
    except IndexError:
        pass

    try:
        q.apply_swap(-1, 1)
        return False
    except IndexError:
        pass

    return True


def test_apply_swap_twice_identity() -> bool:
    q = NQubitSystem(3)
    state = [0.0] * 8
    state[5] = 1.0  # |101>
    q.set_value(state)
    q.apply_swap(0, 2)
    q.apply_swap(0, 2)
    return compare_lists(q.state, state)


def test_apply_swap_non_adjacent_three_qubits() -> bool:
    q = NQubitSystem(3)
    state = [0.0] * 8
    state[4] = 1.0  # |100>
    q.set_value(state)
    q.apply_swap(0, 2)
    expected = [0.0] * 8
    expected[1] = 1.0  # |001>
    return compare_lists(q.state, expected)


def test_measure() -> bool:
    q = NQubitSystem(2)

    # Test 1: |00> always measures "00"
    q.set_value([1.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        result = q.measure()
        if result != "00":
            return False
        if not compare_lists(q.state, [1.0, 0.0, 0.0, 0.0]):
            return False

    # Test 2: |11> always measures "11"
    q.set_value([0.0, 0.0, 0.0, 1.0])
    for _ in range(5):
        result = q.measure()
        if result != "11":
            return False
        if not compare_lists(q.state, [0.0, 0.0, 0.0, 1.0]):
            return False

    # Test 3: Superposition of |00> and |10>
    inv_sqrt2 = 1.0 / (2**0.5)
    for _ in range(20):
        q.set_value([inv_sqrt2, 0.0, inv_sqrt2, 0.0])
        result = q.measure()
        if result not in ["00", "10"]:
            return False
        if result == "00" and not compare_lists(q.state, [1.0, 0.0, 0.0, 0.0]):
            return False
        if result == "10" and not compare_lists(q.state, [0.0, 0.0, 1.0, 0.0]):
            return False

    return True


def test_measure_collapse_superposition_three_qubits() -> bool:
    inv_sqrt2 = 1.0 / (2**0.5)
    for _ in range(10):
        q = NQubitSystem(3)
        state = [0.0] * 8
        state[0] = inv_sqrt2  # |000>
        state[7] = inv_sqrt2  # |111>
        q.set_value(state)
        result = q.measure()
        if result not in ["000", "111"]:
            return False
        if result == "000" and not compare_lists(q.state, [1.0] + [0.0] * 7):
            return False
        if result == "111" and not compare_lists(q.state, [0.0] * 7 + [1.0]):
            return False
    return True


def test_measure_returns_correct_length() -> bool:
    q = NQubitSystem(4)
    q.set_value([1.0] + [0.0] * 15)
    result = q.measure()
    if len(result) != 4:
        return False
    return all(ch in ("0", "1") for ch in result)


def run_tests() -> None:
    tests = [
        ("test_set_value", test_set_value),
        ("test_set_value_rejects_non_list", test_set_value_rejects_non_list),
        ("test_init_invalid_num_qubits", test_init_invalid_num_qubits),
        ("test_get_value_vector", test_get_value_vector),
        ("test_apply_not", test_apply_not),
        ("test_apply_not_three_qubits_mapping", test_apply_not_three_qubits_mapping),
        ("test_apply_not_big_endian_mapping", test_apply_not_big_endian_mapping),
        ("test_apply_h", test_apply_h),
        ("test_apply_h_then_h_identity", test_apply_h_then_h_identity),
        ("test_apply_z", test_apply_z),
        ("test_apply_z_hzh_equals_x", test_apply_z_hzh_equals_x),
        ("test_apply_cnot", test_apply_cnot),
        ("test_apply_cnot_control_off", test_apply_cnot_control_off),
        ("test_apply_cnot_negative_index", test_apply_cnot_negative_index),
        ("test_apply_cnot_three_qubits", test_apply_cnot_three_qubits),
        ("test_apply_cnot_creates_bell_state", test_apply_cnot_creates_bell_state),
        ("test_apply_swap", test_apply_swap),
        ("test_apply_swap_twice_identity", test_apply_swap_twice_identity),
        ("test_apply_swap_non_adjacent_three_qubits", test_apply_swap_non_adjacent_three_qubits),
        ("test_measure", test_measure),
        ("test_measure_collapse_superposition_three_qubits", test_measure_collapse_superposition_three_qubits),
        ("test_measure_returns_correct_length", test_measure_returns_correct_length),
    ]

    for name, fn in tests:
        try:
            result = fn()
        except Exception as e:
            print(f"Exception on {name}:", e)
            result = False

        print(f"{name}: {'PASS' if result else 'FAIL'}")

# %% [markdown]
# ## Task 3: Implement `NQubitSystem`

# %%
## TODO: Implement NQubitSystem Here

class NQubitSystem(QubitSystem):
    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")
        super().__init__(num_qubits=num_qubits)
    
    def set_value(self, new_state) -> None:
        if isinstance(new_state, list) is False:
            raise ValueError("State must be provided as a list")

        required_length = 2 ** self._num_qubits
        if len(new_state) != required_length:
            raise ValueError(
                f"State vector must have length {required_length}, got {len(new_state)}"
            )

        total_probability = 0.0
        for amplitude in new_state:
            total_probability += abs(amplitude) ** 2

        if total_probability < 0.99 or total_probability > 1.01:
            raise ValueError(
                f"State vector not normalized: sum of |amplitude|^2 = {total_probability}"
            )

        self.state = []
        for amplitude in new_state:
            self.state.append(float(amplitude))


    def get_value_braket(self) -> str:
        output_terms = []

        for index in range(len(self.state)):
            amplitude = self.state[index]

            if abs(amplitude) <= 1e-10:
                continue

            basis_string = format(index, f"0{self._num_qubits}b")
            magnitude = abs(amplitude)

            if amplitude < 0:
                sign = "-"
            else:
                sign = "+"

            if len(output_terms) == 0:
                if amplitude < 0:
                    output_terms.append(f"-{magnitude:.4f}|{basis_string}>")
                else:
                    output_terms.append(f"{magnitude:.4f}|{basis_string}>")
            else:
                output_terms.append(f" {sign} {magnitude:.4f}|{basis_string}>")

        if len(output_terms) == 0:
            return "0"

        return "".join(output_terms)


    def get_value_vector(self) -> list[float]:
        copied_state = []
        for value in self.state:
            copied_state.append(value)
        return copied_state


    def apply_not(self, qubit_index: int) -> None:
        self._check_index(qubit_index)

        bit_position = self._num_qubits - 1 - qubit_index
        mask = 1 << bit_position

        updated_state = self.state.copy()

        for index in range(len(self.state)):
            if (index & mask) == 0:
                flipped_index = index | mask

                temp = updated_state[index]
                updated_state[index] = updated_state[flipped_index]
                updated_state[flipped_index] = temp

        self.state = updated_state


    def apply_h(self, qubit_index: int) -> None:
        self._check_index(qubit_index)

        bit_position = self._num_qubits - 1 - qubit_index
        mask = 1 << bit_position
        scale = 1.0 / math.sqrt(2)

        updated_state = self.state.copy()

        for index in range(len(self.state)):
            if (index & mask) == 0:
                partner_index = index | mask

                amp_zero = self.state[index]
                amp_one = self.state[partner_index]

                updated_state[index] = scale * (amp_zero + amp_one)
                updated_state[partner_index] = scale * (amp_zero - amp_one)

        self.state = updated_state


    def apply_z(self, qubit_index: int) -> None:
        self._check_index(qubit_index)

        bit_position = self._num_qubits - 1 - qubit_index
        mask = 1 << bit_position

        updated_state = self.state.copy()

        for index in range(len(self.state)):
            if (index & mask) != 0:
                updated_state[index] = -updated_state[index]

        self.state = updated_state


    def apply_cnot(self, control: int, target: int) -> None:
        self._check_pair(control, target)

        if control == target:
            raise IndexError("Control and target must be different")

        control_mask = 1 << (self._num_qubits - 1 - control)
        target_mask = 1 << (self._num_qubits - 1 - target)

        updated_state = self.state.copy()

        for index in range(len(self.state)):
            control_bit_is_one = (index & control_mask) != 0
            target_bit_is_zero = (index & target_mask) == 0

            if control_bit_is_one and target_bit_is_zero:
                flipped_index = index | target_mask

                temp = updated_state[index]
                updated_state[index] = updated_state[flipped_index]
                updated_state[flipped_index] = temp

        self.state = updated_state


    def apply_swap(self, i: int, j: int) -> None:
        self._check_pair(i, j)

        if i == j:
            raise IndexError("Swap indices must be different")

        mask_i = 1 << (self._num_qubits - 1 - i)
        mask_j = 1 << (self._num_qubits - 1 - j)

        updated_state = self.state.copy()

        for index in range(len(self.state)):
            bit_i_is_one = (index & mask_i) != 0
            bit_j_is_one = (index & mask_j) != 0

            if bit_i_is_one != bit_j_is_one:
                swapped_index = index ^ (mask_i | mask_j)

                if index < swapped_index:
                    temp = updated_state[index]
                    updated_state[index] = updated_state[swapped_index]
                    updated_state[swapped_index] = temp

        self.state = updated_state


    def measure(self) -> str:
        probabilities = []

        for amplitude in self.state:
            probabilities.append(abs(amplitude) ** 2)

        random_value = random.random()
        cumulative_probability = 0.0
        chosen_index = 0

        for index in range(len(probabilities)):
            cumulative_probability += probabilities[index]

            if random_value <= cumulative_probability:
                chosen_index = index
                break

        collapsed_state = [0.0] * len(self.state)
        collapsed_state[chosen_index] = 1.0
        self.state = collapsed_state

        return format(chosen_index, f"0{self._num_qubits}b")


run_tests()


# %% [markdown]
# ## Oracles:
# 
# 
# In class we've discussed two quantum oracles: **BernVaz** and **Archimedes**. Notice how they share a common pattern:
# 
# 1. **State Preparation**: Both oracles expect the system to be in a specific configuration. BernVaz, for example, expects all qubits must be in a superposition state (e.g., created by applying Hadamard gates to all qubits).
# 2. **Oracle Probing**: Once prepared, you apply the oracle itself, which transforms the superposed state according to the code(s) it encodes.
# 3. **Post-Processing**: After the oracle runs, the system requires additional quantum operations to extract useful information. In the case of BernVaz, this means applying Hadamard gates to all qubits again to collapse the superposition in a meaningful way.
# 
# Notice how both oracles follow this same high-level workflow, even though they encode different functions. This is precisely the kind of scenario where **object-oriented programming** shines.
# 
# ## Interfaces: Capturing Shared Behavior
# 
# In OOP, an **interface** (called an abstract base class in python) defines a contract: "Any object that implements this interface must support these operations." Interfaces let you write code that works with *any* object following that contract, without needing to know the specific details of each implementation.
# 
# In our case, you might imagine defining an `Oracle` interface that specifies:
# 
# - An oracle must be able to prepare the system
# - An oracle must be able to apply itself to a system
# - An oracle must be able to post-process the result
# 
# Both `BernVazOracle` and `ArchimedesOracle` could implement this interface, each providing their own specific logic while adhering to the same structure. This way, future code could work with either oracle.
# 
# This pattern is powerful because it lets you **abstract away the differences** (number of qubits, specific gate sequences) while **capturing the similarities** (the three-step workflow). As quantum algorithms grow more complex, this kind of abstraction will help you manage complexity and extend your code without rewriting it.
# 
# # `Oracle` Abstract Base Class
# 
# Implement `Oracle` as an abstract base class (an `ABC` from the Python library `abc`) that defines the interface for quantum oracles (this means `Oracle` should consist entirely of `@abstractmethod`s). It is up to concrete oracle implementations to inherit from this class and provide concrete implementations of the three abstract methods.
# 
# 
# | Function | Behavior |
# | :-- | :-- |
# | `__init__(codes: list[str])` | Initializes the oracle with a list of strings representing the 3-bit codes that define the oracle's behavior. Store these codes for use in the `probe()` method. |
# | `pre_probe(system: NQubitSystem) -> None` | Prepares the quantum system into the state that the oracle expects. This typically involves creating an equal superposition across all qubits. Raise an `IndexError` if `system.num_qubits` is not equal to 4. |
# | `probe(system: NQubitSystem) -> None` | Applies the oracle transformation to the system. The oracle uses the stored codes to determine how to transform each basis state. Raise an `IndexError` if `system.num_qubits` is not equal to 4. |
# | `post_probe(system: NQubitSystem) -> None` | Performs post-processing on the system after the oracle has been applied. This typically involves applying additional gates (e.g., Hadamard gates) to extract meaningful information from the superposed state. Raise an `IndexError` if `system.num_qubits` is not equal to 4. |
# 
# ## Task 4: Implement `Oracle` as an abstract base class

# %%
## TODO: Oracle code below

class Oracle(ABC):
    def __init__(self, codes: list[str]):
        self.codes = codes

    @abstractmethod
    def pre_probe(self, system: "NQubitSystem") -> None:
        raise NotImplementedError

    @abstractmethod
    def probe(self, system: "NQubitSystem") -> None:
        raise NotImplementedError

    @abstractmethod
    def post_probe(self, system: "NQubitSystem") -> None:
        raise NotImplementedError


# %% [markdown]
# ## Task 5: Implement the `BernVaz` oracle as a child of `Oracle`
# ![bern vaz oracle](https://www.classes.cs.uchicago.edu/archive/2026/winter/22880-1/assigns/week5/figs/bernvazoraclealg.png)
# 
# **Note**: We will only test your code with code lists containing a single 3 bit code (e.g., `['011']`). It is up to you to decide how to handle the case of lists containing multiple codes; you may choose to ignore that case or throw an `IndexError` exception.

# %%
## TODO: BernVaz code below

class BernVaz(Oracle):
    def __init__(self, codes: list[str]):
        super().__init__(codes=codes)

    def _validate(self, system: "NQubitSystem") -> str:
        if len(self.codes) != 1:
            raise IndexError("BernVaz expects exactly one code")
        code = self.codes[0]
        if len(code) != 3:
            raise ValueError("BernVaz expects a 3-bit code")
        for ch in code:
            if ch not in ("0", "1"):
                raise ValueError("Code must be binary")
        if system.num_qubits != 4:
            raise IndexError("BernVaz expects a 4-qubit system")
        return code

    def pre_probe(self, system: "NQubitSystem") -> None:
        self._validate(system)
        # Apply H to ALL qubits
        for i in range(system.num_qubits):
            system.apply_h(i)

    # A bunch of CNOTs
    def probe(self, system: "NQubitSystem") -> None:
        code = self._validate(system)
        response_index = system.num_qubits - 1
        for qi, bit in enumerate(code):
            if bit == "1":
                system.apply_cnot(qi, response_index)

    def post_probe(self, system: "NQubitSystem") -> None:
        self._validate(system)
        # Apply H to ALL qubits
        for i in range(system.num_qubits):
            system.apply_h(i)

# %% [markdown]
# ## Task 6: Implement the `Archimedes` oracle as a child of `Oracle`
# ![archimedes oracle](https://www.classes.cs.uchicago.edu/archive/2026/winter/22880-1/assigns/week5/figs/archoraclealg.png)
# 

# %%
# ## TODO: Archimedes code below

class Archimedes(Oracle):
    def __init__(self, codes: list[str]):
        super().__init__(codes=codes)

    def _validate(self, system: "NQubitSystem") -> list[str]:
        if system.num_qubits != 4:
            raise IndexError("Archimedes expects a 4-qubit system")
        if len(self.codes) == 0:
            raise ValueError("Archimedes expects at least one code")
        for code in self.codes:
            if len(code) != 3:
                raise ValueError("Each code must be a 3-bit string")
            for ch in code:
                if ch not in ("0", "1"):
                    raise ValueError("Code must be binary")
        return self.codes

    def pre_probe(self, system: "NQubitSystem") -> None:
        self._validate(system)
        # Apply H to ALL qubits
        for i in range(system.num_qubits):
            system.apply_h(i)


    # Set: (000 010 110). We make a truth table of 0000 -> 0001. 0001 -> 0000. and so on. 
    def probe(self, system: "NQubitSystem") -> None:
        valid_codes = self._validate(system)

        # target is the last qubit -> least significant bit of the basis index
        target_mask = 1

        allowed_inputs = set()
        for code in valid_codes:
            allowed_inputs.add(int(code, 2))

        new_state = system.state.copy()

        for idx in range(len(system.state)):
            # read the first 3 bits by dropping the last bit
            input_value = idx >> 1

            # if x is in the set, flip the target bit
            if input_value in allowed_inputs:
                flipped_idx = idx ^ target_mask

                # swap amplitudes once 
                if idx < flipped_idx:
                    temp = new_state[idx]
                    new_state[idx] = new_state[flipped_idx]
                    new_state[flipped_idx] = temp

        system.state = new_state


    def post_probe(self, system: "NQubitSystem") -> None:
        self._validate(system)
        # Apply H ONLY to the 3 input qubits (top wires)
        for i in range(system.num_qubits - 1):
            system.apply_h(i)

# %% [markdown]
# ## Task 7: Test Your Oracles:
# 
# Finally, write tests to verify that your `Oracle` abstract base class and concrete oracle implementations work correctly. Implement the following:
# 
# 
# | Test Name | Description |
# | :-- | :-- |
# | `test_oracle_is_abstract` | Verify that attempting to instantiate `Oracle` directly raises a `TypeError`. Since `Oracle` is an abstract base class, it should not be possible to create an instance of it without implementing all abstract methods. |
# | `test_bernvaz_is_oracle` | Verify that `BernVazOracle` is a subclass of `Oracle` and can be instantiated with a valid list of 3-bit codes. |
# | `test_archimedes_is_oracle` | Verify that `ArchimedesOracle` is a subclass of `Oracle` and can be instantiated with a valid list of 3-bit codes. |
# | `test_bernvaz` | Create a 4-qubit `NQubitSystem`, prepare it using `BernVazOracle.pre_probe()`, apply the oracle using `BernVazOracle.probe()`, perform post-processing with `BernVazOracle.post_probe()`, then verify that the final state matches your expected result. Test with multiple different code lists to ensure correctness. |
# | `test_archimedes` | Create a 4-qubit `NQubitSystem`, prepare it using `ArchimedesOracle.pre_probe()`, apply the oracle using `ArchimedesOracle.probe()`, perform post-processing with `ArchimedesOracle.post_probe()`, then verify that the final state matches your expected result. Test with multiple different code lists to ensure correctness. |
# 
# 
# 

# %%
## TODO: Oracle Tests

def _apply_h_to_state(state: list[float], num_qubits: int, qubit_index: int) -> list[float]:
    bit_position = num_qubits - 1 - qubit_index
    mask = 1 << bit_position
    scale = 1.0 / (2 ** 0.5)
    new_state = state.copy()
    for index in range(len(state)):
        if (index & mask) == 0:
            partner_index = index | mask
            amp_zero = state[index]
            amp_one = state[partner_index]
            new_state[index] = scale * (amp_zero + amp_one)
            new_state[partner_index] = scale * (amp_zero - amp_one)
    return new_state


def _apply_cnot_to_state(state: list[float], num_qubits: int, control: int, target: int) -> list[float]:
    control_mask = 1 << (num_qubits - 1 - control)
    target_mask = 1 << (num_qubits - 1 - target)
    new_state = state.copy()
    for index in range(len(state)):
        control_bit_is_one = (index & control_mask) != 0
        target_bit_is_zero = (index & target_mask) == 0
        if control_bit_is_one and target_bit_is_zero:
            flipped_index = index | target_mask
            if index < flipped_index:
                temp = new_state[index]
                new_state[index] = new_state[flipped_index]
                new_state[flipped_index] = temp
    return new_state


def _expected_bernvaz_probe_state(secret_code: str, num_qubits: int) -> list[float]:
    state = _expected_h_all_state(num_qubits)
    response_index = num_qubits - 1
    for qi, bit in enumerate(secret_code):
        if bit == "1":
            state = _apply_cnot_to_state(state, num_qubits, qi, response_index)
    return state


def _expected_bernvaz_state(secret_code: str) -> list[float]:
    """Expected final state for BernVaz with diagram: H all -> CNOT oracle -> H all."""
    num_qubits = len(secret_code) + 1
    state = _expected_bernvaz_probe_state(secret_code, num_qubits)
    for i in range(num_qubits):
        state = _apply_h_to_state(state, num_qubits, i)
    return state


def _expected_archimedes_state(n: int, f) -> list[float]:
    """Expected final state for Archimedes with diagram: H all -> phase oracle -> H on inputs only."""
    total_states = 2 ** (n + 1)
    expected = [0.0 for _ in range(total_states)]
    scale = 1.0 / (2 ** ((2 * n + 1) / 2))

    for y in range(2 ** n):
        phase_sum = 0
        for x in range(2 ** n):
            parity = bin(x & y).count("1") % 2
            phase = -1 if (f(x) + parity) % 2 == 1 else 1
            phase_sum += phase

        amp = scale * phase_sum
        base = y << 1
        expected[base] = amp
        expected[base | 1] = amp  # response qubit in |+>
    return expected


def _expected_h_all_state(num_qubits: int) -> list[float]:
    amp = 1.0 / (2 ** (num_qubits / 2))
    return [amp for _ in range(2 ** num_qubits)]


def _expected_phase_kickback_state(code: str, num_qubits: int) -> list[float]:
    response_mask = 1
    amp = 1.0 / (2 ** (num_qubits / 2))
    expected = [amp for _ in range(2 ** num_qubits)]
    for idx in range(len(expected)):
        if (idx & response_mask) != 0:
            continue
        dot = 0
        for qi, bit in enumerate(code):
            if bit == "1":
                bit_position = num_qubits - 1 - qi
                if (idx & (1 << bit_position)) != 0:
                    dot ^= 1
        if dot == 1:
            paired = idx | response_mask
            expected[idx] *= -1
            expected[paired] *= -1
    return expected


def test_oracle_is_abstract() -> bool:
    try:
        Oracle(["000"])
        return False
    except TypeError:
        return True


def test_bernvaz_is_oracle() -> bool:
    if not issubclass(BernVaz, Oracle):
        return False
    try:
        _ = BernVaz(["101"])
    except Exception:
        return False
    return True


def test_archimedes_is_oracle() -> bool:
    if not issubclass(Archimedes, Oracle):
        return False
    try:
        _ = Archimedes(["010"])
    except Exception:
        return False
    return True


def test_bernvaz_pre_probe_state() -> bool:
    system = NQubitSystem(4)
    oracle = BernVaz(["101"])
    oracle.pre_probe(system)
    expected = _expected_h_all_state(4)
    return compare_lists(system.state, expected)


def test_bernvaz_probe_state() -> bool:
    code = "101"
    system = NQubitSystem(4)
    oracle = BernVaz([code])
    oracle.pre_probe(system)
    oracle.probe(system)
    expected = _expected_bernvaz_probe_state(code, 4)
    return compare_lists(system.state, expected)


def test_bernvaz_post_probe_state() -> bool:
    code = "101"
    system = NQubitSystem(4)
    oracle = BernVaz([code])
    oracle.pre_probe(system)
    oracle.probe(system)
    oracle.post_probe(system)
    expected = _expected_bernvaz_state(code)
    return compare_lists(system.state, expected)


def _expected_archimedes_probe_state(codes: list[str], num_qubits: int) -> list[float]:
    """
    Expected state after:
      - H on all qubits
      - Archimedes oracle (bit-flip on response if input in set)
    """
    amp = 1.0 / (2 ** (num_qubits / 2))
    state = [amp for _ in range(2 ** num_qubits)]

    allowed_inputs = {int(code, 2) for code in codes}
    response_mask = 1  # last qubit (LSB)

    new_state = state.copy()

    for idx in range(len(state)):
        input_value = idx >> 1
        if input_value in allowed_inputs:
            flipped = idx ^ response_mask
            if idx < flipped:
                temp = new_state[idx]
                new_state[idx] = new_state[flipped]
                new_state[flipped] = temp

    return new_state


def _expected_archimedes_final_state(codes: list[str]) -> list[float]:
    """
    Expected final state after:
      - H on all
      - Archimedes oracle
      - H on input qubits only
    """
    num_qubits = 4
    state = _expected_archimedes_probe_state(codes, num_qubits)

    # apply H to input qubits only (0,1,2)
    for i in range(3):
        state = _apply_h_to_state(state, num_qubits, i)

    return state


def test_archimedes_pre_probe() -> bool:
    system = NQubitSystem(4)
    oracle = Archimedes(["010", "111"])
    oracle.pre_probe(system)

    expected = _expected_h_all_state(4)
    return compare_lists(system.state, expected)


def test_archimedes_probe() -> bool:
    codes = ["010", "111"]
    system = NQubitSystem(4)
    oracle = Archimedes(codes)

    oracle.pre_probe(system)
    oracle.probe(system)

    expected = _expected_archimedes_probe_state(codes, 4)
    return compare_lists(system.state, expected)


def test_archimedes_post_probe() -> bool:
    codes = ["010", "111"]
    system = NQubitSystem(4)
    oracle = Archimedes(codes)

    oracle.pre_probe(system)
    oracle.probe(system)
    oracle.post_probe(system)

    expected = _expected_archimedes_final_state(codes)
    return compare_lists(system.state, expected)


def test_bernvaz_invalid_code_length() -> bool:
    try:
        system = NQubitSystem(4)
        oracle = BernVaz(["10"])
        oracle.pre_probe(system)
        return False
    except ValueError:
        return True


def test_bernvaz_invalid_code_characters() -> bool:
    try:
        system = NQubitSystem(4)
        oracle = BernVaz(["10A"])
        oracle.pre_probe(system)
        return False
    except ValueError:
        return True


def test_bernvaz_multiple_codes_rejected() -> bool:
    try:
        system = NQubitSystem(4)
        oracle = BernVaz(["010", "111"])
        oracle.pre_probe(system)
        return False
    except IndexError:
        return True


def test_bernvaz_wrong_qubits() -> bool:
    try:
        system = NQubitSystem(3)
        oracle = BernVaz(["010"])
        oracle.pre_probe(system)
        return False
    except IndexError:
        return True


def test_archimedes_empty_codes() -> bool:
    try:
        system = NQubitSystem(4)
        oracle = Archimedes([])
        oracle.pre_probe(system)
        return False
    except ValueError:
        return True


def test_archimedes_invalid_code_length() -> bool:
    try:
        system = NQubitSystem(4)
        oracle = Archimedes(["1"])
        oracle.pre_probe(system)
        return False
    except ValueError:
        return True


def test_archimedes_invalid_code_characters() -> bool:
    try:
        system = NQubitSystem(4)
        oracle = Archimedes(["01X"])
        oracle.pre_probe(system)
        return False
    except ValueError:
        return True


def test_archimedes_wrong_qubits() -> bool:
    try:
        system = NQubitSystem(5)
        oracle = Archimedes(["010"])
        oracle.pre_probe(system)
        return False
    except IndexError:
        return True


def test_bernvaz() -> bool:
    codes = ["101"]
    system = NQubitSystem(4)
    oracle = BernVaz(codes)

    oracle.pre_probe(system)
    oracle.probe(system)
    oracle.post_probe(system)

    expected = _expected_bernvaz_state(codes[0])
    return compare_lists(system.state, expected)


def test_archimedes() -> bool:
    codes = ["010", "111"]
    system = NQubitSystem(4)
    oracle = Archimedes(codes)

    oracle.pre_probe(system)
    oracle.probe(system)
    oracle.post_probe(system)

    expected = _expected_archimedes_final_state(codes)
    return compare_lists(system.state, expected)


def run_oracle_tests() -> None:
    tests = [
        ("test_oracle_is_abstract", test_oracle_is_abstract),
        ("test_bernvaz_is_oracle", test_bernvaz_is_oracle),
        ("test_archimedes_is_oracle", test_archimedes_is_oracle),
        ("test_bernvaz_pre_probe_state", test_bernvaz_pre_probe_state),
        ("test_bernvaz_probe_state", test_bernvaz_probe_state),
        ("test_bernvaz_post_probe_state", test_bernvaz_post_probe_state),
        ("test_archimedes_pre_probe", test_archimedes_pre_probe),
        ("test_archimedes_probe", test_archimedes_probe),
        ("test_archimedes_post_probe", test_archimedes_post_probe),
        ("test_bernvaz_invalid_code_length", test_bernvaz_invalid_code_length),
        ("test_bernvaz_invalid_code_characters", test_bernvaz_invalid_code_characters),
        ("test_bernvaz_multiple_codes_rejected", test_bernvaz_multiple_codes_rejected),
        ("test_bernvaz_wrong_qubits", test_bernvaz_wrong_qubits),
        ("test_archimedes_empty_codes", test_archimedes_empty_codes),
        ("test_archimedes_invalid_code_length", test_archimedes_invalid_code_length),
        ("test_archimedes_invalid_code_characters", test_archimedes_invalid_code_characters),
        ("test_archimedes_wrong_qubits", test_archimedes_wrong_qubits),
        ("test_bernvaz", test_bernvaz),
        ("test_archimedes", test_archimedes),
    ]

    for name, fn in tests:
        try:
            result = fn()
        except Exception as e:
            print(f"Exception on {name}:", e)
            result = False
        print(f"{name}: {'PASS' if result else 'FAIL'}")


run_oracle_tests()

# %% [markdown]
# # Submission
# Congratulations on completing the lab!
# Make sure you:
# 
# 
# 1.   Test all of your functions by calling them at least once.
# 2.   Download your lab as a **python** `.py` script (NOT an `.ipynb` file):
#       
#       ```File -> Download -> Download .py```
# 
# 3.   Rename the downloaded file to `Lab5Answers.py`.
# 4.   Upload the `Lab5Answers.py` file to Gradescope.
# 5.   Ensure the autograder runs successfully.

# %% [markdown]
# 


