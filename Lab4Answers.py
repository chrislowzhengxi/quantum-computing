# %% [markdown]
# # Lab 4: Simulating a single qubit system
# In this assignment, you’ll build a small (but real) quantum state simulator in Python while practicing core software engineering skills. You’ll implement a reusable `QubitSystem` interface (using an abstract base class) that supports initializing and inspecting quantum states and applying a handful of standard quantum gates, then create a concrete `SingleQubitSystem` implementation that inherits from it. Along the way, you’ll strengthen your understanding of how quantum states can be represented (bra-ket vs. state vectors), how **object-oriented** design helps you structure and extend code cleanly, and how test-driven development (TDD) can guide you to implement correct behavior incrementally with confidence.
# 

# %% [markdown]
# # Object-oriented programming (OOP)
# 
# Object-oriented programming is a way to organize code around “objects” that bundle **data** (attributes) with **behavior** (methods). A core OOP idea is to define a general blueprint for a family of related things, then create more specific versions by extending that blueprint. This helps you reuse code, keep responsibilities clear, and make it easier to add new features without rewriting existing logic.
# 
# ### Base classes and shared behavior
# 
# A common pattern is to create a **base class** (sometimes called an interface) that defines what all related objects should be able to do. In the example below, `Polygon` represents “any shape with sides.” Because every polygon has a number of sides, we can implement `print_number_of_sides()` once in `Polygon` and automatically reuse it in every child class.
# 
# ### Abstract methods and specialization
# 
# Some behaviors depend on the specific type of polygon. For instance, the formula for area is different for triangles and rectangles, so `calculate_area()` is declared as an abstract method in `Polygon`. That means `Polygon` is promising “every polygon can calculate an area,” but it requires each subclass to provide the actual formula.
# 
# ### Inheritance and `super()`
# 
# When you write a child class, you often still want to run some of the “standard setup” code that the parent class already provides. Python’s `super()` lets you call that parent-class code from inside your child class, instead of copying it.
# 
# For example, in `Triangle`, we call `super().__init__(num_sides=3)` to run the `Polygon` constructor so the triangle is initialized with the correct number of sides. Then we add triangle-specific fields like `base` and `height`.
# 

# %%
from abc import ABC, abstractmethod

class Polygon(ABC):
    def __init__(self, num_sides: int):
        self.num_sides = num_sides

    def print_number_of_sides(self) -> str:
        return f"This polygon has {self.num_sides} sides."

    @abstractmethod
    def calculate_area(self) -> float:
        """Return the area of this polygon."""
        raise NotImplementedError


class Triangle(Polygon):
    def __init__(self, base: float, height: float):
        super().__init__(num_sides=3)
        self.base = base
        self.height = height

    def calculate_area(self) -> float:
        return 0.5 * self.base * self.height


class Rectangle(Polygon):
    def __init__(self, width: float, height: float):
        super().__init__(num_sides=4)
        self.width = width
        self.height = height

    def calculate_area(self) -> float:
        return self.width * self.height


# %%
# We can now instantiate (create an instance of) a triangle:
t = Triangle(10, 4)

# Compute its area
print(t.calculate_area())

# We can also instantiate a rectangle:
r = Rectangle(10, 4)

# It has a different area!
print(r.calculate_area())

# %% [markdown]
# In this assignment, you’ll use the same design approach: define a general (abstract) `QubitSystem` with shared behaviors where possible, then implement a specific `SingleQubitSystem` that inherits from it and fills in the details (or restricts operations that don’t make sense).
# 

# %% [markdown]
# # Test Driven Development
# 
# Test-driven development (TDD) means you write a small test first, watch it fail (because the feature isn’t implemented yet), then write the minimum code to make it pass, and finally clean up your design without changing behavior. This workflow helps you lock in expected behavior early and catch regressions as you refactor your classes.
# 
# The general approach for a test is:
# - **Arrange**: create the instances of the object you are testing
# - **Act**: perform actions or operations on the instance of the object
# - **Assert**: validate the result of the actions being tested
# 
# ### What we’re testing
# 
# For the `Triangle` subclass, we’ll write two tests:
# 
# - `test_calculate_area`: verifies `calculate_area()` returns the correct numeric result for known inputs.
# - `test_print_number_of_sides`: verifies `print_number_of_sides()` prints the expected text (since it’s implemented in the parent `Polygon` class but should work when called on a `Triangle`).
# 
# Each test will return `True` if behavior is correct and `False` otherwise, so you can run them without any testing framework.
# 
# ### Test 1: `test_calculate_area`
# 
# This test constructs a triangle with a known base and height, then checks whether the computed area matches what you expect.
# 

# %%
def test_calculate_area() -> bool:
    # Arrange
    t = Triangle(base=10, height=4)

    # Act
    area = t.calculate_area()

    # Assert
    expected = 20.0  # 0.5 * 10 * 4
    return area == expected


# %% [markdown]
# Now let's write a test for `print_number_of_sides`.

# %%
def test_print_number_of_sides() -> bool:
    # Arrange
    t = Triangle(base=3, height=5)

    # Act
    printed = t.print_number_of_sides()

    # Assert
    return printed == "This polygon has 3 sides."

# %% [markdown]
# ### Running your tests
# 
# A simple runner can quickly call each test and print whether it passed:

# %%
def run_tests() -> None:
    tests = [
        ("test_calculate_area", test_calculate_area),
        ("test_print_number_of_sides", test_print_number_of_sides),
    ]

    for name, fn in tests:
        result = fn()
        print(f"{name}: {'PASS' if result else 'FAIL'}")

run_tests()


# %% [markdown]
# Oh no! The test for number of sides failed! That means our code must have a bug in it.
# 
# ## Task 0: Ensure our tests pass
# 
# Find and fix the bug in our code above, then run the test suite again to make sure it passes!

# %% [markdown]
# # Assignment Specs:
# Now that you are familiar with OOP we can move on the `QubitSystem`. You will be implementing the following:
# 
# 
# ### `QubitSystem` technical specs
# 
# | Function | Behavior |
# | --- | --- |
# | `set_value(value)` | Sets the system’s quantum state. Accept a list of floats representing the state vector amplitudes in **big-endian** basis order. Implementations should validate that the input represents a valid state for the system size and raise a `ValueError` if not. |
# | `get_value_braket() -> str` | Returns a bra-ket style string representing the current state. This is primarily for readability/debugging, you won't be penalized for stylistic differences or formating. |
# | `get_value_vector() -> list[float]` | Returns the current state vector amplitudes in big-endian basis order. |
# | `apply_not(i: int) -> None` | Applies the NOT gate (Pauli-$X$) to qubit at index $i$, updating the system state. Raise an `IndexError` if $i$ is not a valid qubit index. |
# | `apply_h(i: int) -> None` | Applies the Hadamard gate ($H$) to qubit at index $i$, updating the system state. Raise an `IndexError` if $i$ is not a valid qubit index. |
# | `apply_z(i: int) -> None` | Applies the Pauli-$Z$ gate to qubit at index $i$, updating the system state. Raise an `IndexError` if $i$ is not a valid qubit index. |
# | `apply_cnot(control: int, target: int) -> None` | Applies a controlled-NOT gate with `control` as the control qubit and `target` as the target qubit, updating the system state. Raise an `IndexError` if either index is invalid, or if they are the same. |
# | `apply_swap(i: int, j: int) -> None` | Swaps qubits at indices $i$ and $j$, updating the system state. Raise an `IndexError` if either index is invalid, or if they are the same. |
# | `measure() -> str` | <p>Simulates a measurement of the state of the system and returns one of the possible values as a big-endian string of binary: e.g., if the state $\lvert 00 \rangle$ is being measured the result would always be `'00'`, If the state $\frac{1}{\sqrt{2}} \lvert 00 \rangle + \frac{1}{\sqrt{2}} \lvert 01 \rangle$ is measured half the time we would expect `'00'` the other half we would expect `'01'`.<ul><li>Note: If a system is in a state of superposition before `measure()`, the act of measurement should collapse the superposition.</li><li>Note 2: The output should always have the same number of bits as the system does. For `SingleQubitSystem` the outputs will be `'0'` or `'1'`.</li></ul></p> |
# 
# ### `SingleQubitSystem` technical specs
# 
# `SingleQubitSystem`: A concrete `QubitSystem` implementation for exactly one qubit. It must support initialization, retrieval in both formats, and single-qubit gates (`apply_not`, `apply_h`, `apply_z`) with correct state updates.
# 
# | Function | Behavior |
# | --- | --- |
# | `set_value(value)` | In addition to supporting a list of floats, the `SingleQubitSystem`'s `set_value` will also accept the following strings representing bra-ket states: `"\|0>"`, `"\|1>"`, `"\|+>"` or `"\|->"` |
# | `apply_cnot(control: int, target: int) -> None` | Must always raise an `IndexError` when called on a single-qubit system (since there is no valid pair of qubit indices). |
# | `apply_swap(i: int, j: int) -> None` | Must always raise an `IndexError` when called on a single-qubit system (since there is no valid pair of qubit indices). |

# %% [markdown]
# 
# ## Importing external libraries (don't): 
# You are free to use [`numpy`](https://numpy.org/doc/stable/) to handle all matrix and mathematical operations or you may choose to implement everything yourself as helper functions. Regardless of your decision, we expect all students to know how to programmatically implement matrix multiplication (*hint: potential midterm question*). You may **not** use `qiskit` or **any** other library besides `numpy` as part of your solution.
# **Adding extra import statement _will_ crash the autograder**
# 
# ## Implementation Steps:
# For now *all* of `QubitSystem`'s functions are marked as abstract; it is up to you to decide which ones to implement and which ones to delegate (i.e., only implement in child classes).
# 
# If you decide the functionality is broad enough that ALL qubit systems regardless of number of qubits would be able to use it you should:
# 
# 
# 1.   Implement it in `QubitSystem`
# 2.   Remove the `@abstractmethod` tag
# 3.   Delete the function definition from `SingleQubitSystem` (unless you will be [overloading](https://en.wikipedia.org/wiki/Function_overloading) it. For an example of this see the `__init__` methods)
# 
# If instead you think a particular set of functionality should be implemented by `SingleQubitSystem` you may leave everything as is and go straight to `SingleQubitSystem`.
# 
# Next week's lab will involve implementing an `NQubitSystem` which *will* be able to handle all gates so it is worth the effort to plan out a versatile implementation **now** rather than having to deal with refractoring later.
# 
# Note: if you decide to declare helper functions **make sure to include them within your class declarations**; otherwise, your submission might crash when being autograded.
# 
# For your convenience, we have provided skeleton code to get you started. **Do not rename any of the provided functions.**
# 
# ## Task 1: Before you start implementing, scroll down and review the test suite

# %%
from abc import ABC, abstractmethod
import random
import math

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
# # Test Suite
# We will begin by defining the bare-bones testing suite. **The autograder will expect for these tests to be present in your submission and will evaluate them for correctness; do not change their names.**
# 
# As you progress through the lab be sure to revisit your tests and come up with meaningful ways to evaluate your program's functionality under known circumstances. Your tests should always exit gracefully, even when intentionally triggering errors (e.g., ensuring a `SingleQubitSystem` throws an `IndexError` when a multi-qubit gate is applied): your code should catch the errors rather than letting them propagate.
# 
# You will be graded on your test-suite's coverage in addition to the correctness of your `QubitSystem` and `SingleQubitSystem` classes.
# 
# ## Task 2: Comparing `float`s
# Comparing floating point numbers is trickier than it might seem due to rounding errors. The following exmple shows how even mathematically identical numbers can be interpreted as different due to the way computers handle data:
# 

# %%
a = 0.0
b = 0.0

# Same math, different grouping
for _ in range(1_000_000):
    a += 0.1

b = 0.1 * 1_000_000

print(a)
print(b)
print(a == b)

# %% [markdown]
# To help with this, we have provided with a `compare_lists` function. It compares two lists and, so long as all numbers are within a small margin of each other (conventionally labeled as epsilon $\epsilon$), it will return `True`. We recommend you use it when testing your code.

# %%
def compare_lists(first_list, second_list, eps: float = 1e-3) -> bool:
    import numpy as np
    return bool(np.allclose(first_list, second_list, atol=eps, rtol=0.0))

print([a]==[b])
print(compare_lists([a], [b]))

# %% [markdown]
# # Task 3: Implement `set_value`
# We've provided a minimal test case for `set_value`. Scroll up to the class definitions, implement the barebones functionality (storing a properly formatted list) required to pass the test, then run the cells containing your class definitions and testing suite again.
# 
# The autograder will evaluate each of the test cases below for coverage so you must add multiple cycles of arrange-act-assert within each case. **Do not rename these test cases, any helper functions must be defined _within_ an existing test case**

# %%
def test_set_value() -> bool:
    # Arrange: Set up a new SingleQubit to evaluate
    q = SingleQubitSystem()

    # Test 1: Basic list input [1, 0]
    q.set_value([1.0, 0.0])
    if not compare_lists(q.state, [1.0, 0.0]):
        return False

    # Test 2: List input [0, 1]
    q.set_value([0.0, 1.0])
    if not compare_lists(q.state, [0.0, 1.0]):
        return False

    # Test 3: Superposition state
    inv_sqrt2 = 1.0 / (2**0.5)
    q.set_value([inv_sqrt2, inv_sqrt2])
    if not compare_lists(q.state, [inv_sqrt2, inv_sqrt2]):
        return False

    # Test 4: String input |0>
    q.set_value("|0>")
    if not compare_lists(q.state, [1.0, 0.0]):
        return False

    # Test 5: String input |1>
    q.set_value("|1>")
    if not compare_lists(q.state, [0.0, 1.0]):
        return False

    # Test 6: String input |+>
    q.set_value("|+>")
    if not compare_lists(q.state, [inv_sqrt2, inv_sqrt2]):
        return False

    # Test 7: String input |->
    q.set_value("|->")
    if not compare_lists(q.state, [inv_sqrt2, -inv_sqrt2]):
        return False

    # Test 8: Invalid string should raise ValueError
    try:
        q.set_value("|invalid>")
        return False
    except ValueError:
        pass

    # Test 9: Invalid state length should raise ValueError
    try:
        q.set_value([1.0, 0.0, 0.0])
        return False
    except ValueError:
        pass

    # Test 10: Non-normalized state should raise ValueError
    try:
        q.set_value([2.0, 0.0])
        return False
    except ValueError:
        pass

    # Test 11: Arbitrary normalized superposition
    q.set_value([0.6, 0.8])
    if not compare_lists(q.state, [0.6, 0.8]):
        return False
    
    # Test 12: Invalid type should raise ValueError
    try:
        q.set_value(42)
        return False
    except ValueError:
        pass

    return True


def test_get_value_vector() -> bool:
    # Arrange
    q = SingleQubitSystem()
    
    # Test 1: Get |0>
    q.set_value([1.0, 0.0])
    vec = q.get_value_vector()
    if not compare_lists(vec, [1.0, 0.0]):
        return False
    
    # Test 2: Get |1>
    q.set_value([0.0, 1.0])
    vec = q.get_value_vector()
    if not compare_lists(vec, [0.0, 1.0]):
        return False
    
    # Test 3: Get |+>
    inv_sqrt2 = 1.0 / (2**0.5)
    q.set_value("|+>")
    vec = q.get_value_vector()
    if not compare_lists(vec, [inv_sqrt2, inv_sqrt2]):
        return False
    
    # Test 4: Vector returned is a copy (not reference)
    q.set_value([1.0, 0.0])
    vec = q.get_value_vector()
    vec[0] = 0.5
    if not compare_lists(q.state, [1.0, 0.0]):
        return False
    
    # Test 5: Get vector after gate operation
    q.set_value([1.0, 0.0])
    q.apply_h(0)
    vec = q.get_value_vector()
    if not compare_lists(vec, [inv_sqrt2, inv_sqrt2]):
        return False
    
    # Test 6: Get vector with arbitrary state
    q.set_value([0.6, 0.8])
    vec = q.get_value_vector()
    if not compare_lists(vec, [0.6, 0.8]):
        return False
    
    return True


def test_apply_not() -> bool:
    # Arrange
    q = SingleQubitSystem()
    
    # Test 1: NOT on |0> gives |1>
    q.set_value([1.0, 0.0])
    q.apply_not(0)
    if not compare_lists(q.state, [0.0, 1.0]):
        return False
    
    # Test 2: NOT on |1> gives |0>
    q.set_value([0.0, 1.0])
    q.apply_not(0)
    if not compare_lists(q.state, [1.0, 0.0]):
        return False
    
    # Test 3: NOT on |+> gives |+> (invariant)
    inv_sqrt2 = 1.0 / (2**0.5)
    q.set_value([inv_sqrt2, inv_sqrt2])
    q.apply_not(0)
    if not compare_lists(q.state, [inv_sqrt2, inv_sqrt2]):
        return False
    
    # Test 4: NOT on |-> gives -|-> (up to global phase)
    q.set_value([inv_sqrt2, -inv_sqrt2])
    q.apply_not(0)
    if not compare_lists(q.state, [-inv_sqrt2, inv_sqrt2]):
        return False
    
    # Test 5: Invalid index should raise IndexError
    try:
        q.apply_not(1)
        return False
    except IndexError:
        pass
    
    # Test 6: Negative index should raise IndexError
    try:
        q.apply_not(-1)
        return False
    except IndexError:
        pass
    
    # Test 7: NOT on arbitrary superposition
    q.set_value([0.6, 0.8])
    q.apply_not(0)
    if not compare_lists(q.state, [0.8, 0.6]):
        return False
    
    # Test 8: Large invalid index should raise IndexError
    try:
        q.apply_not(100)
        return False
    except IndexError:
        pass
    
    return True


def test_apply_h() -> bool:
    # Arrange
    q = SingleQubitSystem()
    inv_sqrt2 = 1.0 / (2**0.5)
    
    # Test 1: H on |0> gives |+>
    q.set_value([1.0, 0.0])
    q.apply_h(0)
    if not compare_lists(q.state, [inv_sqrt2, inv_sqrt2]):
        return False
    
    # Test 2: H on |1> gives |->
    q.set_value([0.0, 1.0])
    q.apply_h(0)
    if not compare_lists(q.state, [inv_sqrt2, -inv_sqrt2]):
        return False
    
    # Test 3: H on |+> gives |0>
    q.set_value([inv_sqrt2, inv_sqrt2])
    q.apply_h(0)
    if not compare_lists(q.state, [1.0, 0.0]):
        return False
    
    # Test 4: H on |-> gives |1>
    q.set_value([inv_sqrt2, -inv_sqrt2])
    q.apply_h(0)
    if not compare_lists(q.state, [0.0, 1.0]):
        return False
    
    # Test 5: Invalid index should raise IndexError
    try:
        q.apply_h(1)
        return False
    except IndexError:
        pass
    
    # Test 6: H on arbitrary state
    q.set_value([0.6, 0.8])
    q.apply_h(0)
    expected_alpha = inv_sqrt2 * (0.6 + 0.8)
    expected_beta = inv_sqrt2 * (0.6 - 0.8)
    if not compare_lists(q.state, [expected_alpha, expected_beta]):
        return False
    
    # Test 7: Negative index should raise IndexError
    try:
        q.apply_h(-1)
        return False
    except IndexError:
        pass
    
    return True


def test_apply_z() -> bool:
    # Arrange
    q = SingleQubitSystem()
    inv_sqrt2 = 1.0 / (2**0.5)
    
    # Test 1: Z on |0> gives |0> (invariant)
    q.set_value([1.0, 0.0])
    q.apply_z(0)
    if not compare_lists(q.state, [1.0, 0.0]):
        return False
    
    # Test 2: Z on |1> gives -|1>
    q.set_value([0.0, 1.0])
    q.apply_z(0)
    if not compare_lists(q.state, [0.0, -1.0]):
        return False
    
    # Test 3: Z on |+> gives |->
    q.set_value([inv_sqrt2, inv_sqrt2])
    q.apply_z(0)
    if not compare_lists(q.state, [inv_sqrt2, -inv_sqrt2]):
        return False
    
    # Test 4: Z on |-> gives |+>
    q.set_value([inv_sqrt2, -inv_sqrt2])
    q.apply_z(0)
    if not compare_lists(q.state, [inv_sqrt2, inv_sqrt2]):
        return False
    
    # Test 5: Invalid index should raise IndexError
    try:
        q.apply_z(1)
        return False
    except IndexError:
        pass
    
    # Test 6: Z on arbitrary superposition
    q.set_value([0.6, 0.8])
    q.apply_z(0)
    if not compare_lists(q.state, [0.6, -0.8]):
        return False
    
    # Test 7: Negative index should raise IndexError
    try:
        q.apply_z(-1)
        return False
    except IndexError:
        pass
    
    return True


def test_apply_cnot() -> bool:
    # Single qubit system should always raise IndexError for CNOT
    q = SingleQubitSystem()
    
    # Test 1: CNOT with (0, 0) should raise IndexError
    try:
        q.apply_cnot(0, 0)
        return False
    except IndexError:
        pass
    
    # Test 2: CNOT with (0, 1) should raise IndexError
    try:
        q.apply_cnot(0, 1)
        return False
    except IndexError:
        pass
    
    # Test 3: CNOT with (1, 0) should raise IndexError
    try:
        q.apply_cnot(1, 0)
        return False
    except IndexError:
        pass
    
    # Test 4: CNOT with negative indices should raise IndexError
    try:
        q.apply_cnot(-1, 0)
        return False
    except IndexError:
        pass
    
    # Test 5: CNOT with (1, 1) should raise IndexError
    try:
        q.apply_cnot(1, 1)
        return False
    except IndexError:
        pass
    
    return True


def test_apply_swap() -> bool:
    # Single qubit system should always raise IndexError for SWAP
    q = SingleQubitSystem()
    
    # Test 1: SWAP with (0, 0) should raise IndexError
    try:
        q.apply_swap(0, 0)
        return False
    except IndexError:
        pass
    
    # Test 2: SWAP with (0, 1) should raise IndexError
    try:
        q.apply_swap(0, 1)
        return False
    except IndexError:
        pass
    
    # Test 3: SWAP with negative indices should raise IndexError
    try:
        q.apply_swap(-1, 0)
        return False
    except IndexError:
        pass
    
    # Test 4: SWAP with (1, 0) should raise IndexError
    try:
        q.apply_swap(1, 0)
        return False
    except IndexError:
        pass
    
    return True


def test_measure() -> bool:
    # Arrange
    q = SingleQubitSystem()
    
    # Test 1: Measure definite state |0> always gives "0"
    q.set_value([1.0, 0.0])
    for _ in range(5):
        result = q.measure()
        if result != "0":
            return False
        # State should collapse to |0>
        if not compare_lists(q.state, [1.0, 0.0]):
            return False
    
    # Test 2: Measure definite state |1> always gives "1"
    q.set_value([0.0, 1.0])
    for _ in range(5):
        result = q.measure()
        if result != "1":
            return False
        # State should collapse to |1>
        if not compare_lists(q.state, [0.0, 1.0]):
            return False
    
    # Test 3: Measure |+> gives results and collapses properly
    inv_sqrt2 = 1.0 / (2**0.5)
    q.set_value([inv_sqrt2, inv_sqrt2])
    has_zero = False
    has_one = False
    
    for _ in range(50):
        # Re-set the state to |+> before each measurement since it collapsed
        q.set_value([inv_sqrt2, inv_sqrt2])
        result = q.measure()
        if result == "0":
            has_zero = True
            if not compare_lists(q.state, [1.0, 0.0]):
                return False
        elif result == "1":
            has_one = True
            if not compare_lists(q.state, [0.0, 1.0]):
                return False
        else:
            return False
    
    # Should have observed both outcomes (very unlikely to get all 0s or all 1s in 50 trials)
    if not (has_zero and has_one):
        return False
    
    # Test 4: Result is always "0" or "1" (binary string)
    for _ in range(20):
        q.set_value([inv_sqrt2, inv_sqrt2])
        result = q.measure()
        if result not in ["0", "1"]:
            return False
    
    # Test 5: Measure |-> gives results and collapses properly
    q.set_value([inv_sqrt2, -inv_sqrt2])
    has_zero = False
    has_one = False
    
    for _ in range(50):
        q.set_value([inv_sqrt2, -inv_sqrt2])
        result = q.measure()
        if result == "0":
            has_zero = True
            if not compare_lists(q.state, [1.0, 0.0]):
                return False
        elif result == "1":
            has_one = True
            if not compare_lists(q.state, [0.0, 1.0]):
                return False
        else:
            return False
    
    if not (has_zero and has_one):
        return False
    
    # Test 6: Measure arbitrary superposition (0.6|0> + 0.8|1>)
    q.set_value([0.6, 0.8])
    result = q.measure()
    if result == "0":
        if not compare_lists(q.state, [1.0, 0.0]):
            return False
    elif result == "1":
        if not compare_lists(q.state, [0.0, 1.0]):
            return False
    else:
        return False
    
    return True


def test_measurement_idempotence() -> bool:
    q = SingleQubitSystem()
    inv_sqrt2 = 1.0 / (2**0.5)
    
    # Test 1: After measuring |+>, repeated measures give same result
    q.set_value([inv_sqrt2, inv_sqrt2])
    first = q.measure()
    for _ in range(10):
        result = q.measure()
        if result != first:
            return False
    
    # Test 2: After set_value, measurement can differ
    q.set_value([inv_sqrt2, inv_sqrt2])
    new_result = q.measure()
    if new_result not in ["0", "1"]:
        return False
    
    return True


def test_set_value_defensive_copy() -> bool:
    q = SingleQubitSystem()
    
    # Test 1: Modifying input list doesn't affect internal state
    input_list = [1.0, 0.0]
    q.set_value(input_list)
    input_list[0] = 0.5
    if not compare_lists(q.state, [1.0, 0.0]):
        return False
    
    # Test 2: Another case with superposition
    inv_sqrt2 = 1.0 / (2**0.5)
    input_list = [inv_sqrt2, inv_sqrt2]
    q.set_value(input_list)
    input_list[0] = 0.0
    if not compare_lists(q.state, [inv_sqrt2, inv_sqrt2]):
        return False
    
    return True


def test_gate_normalization_preservation() -> bool:
    q = SingleQubitSystem()
    inv_sqrt2 = 1.0 / (2**0.5)
    
    def is_normalized(state):
        norm_sq = sum(abs(amp)**2 for amp in state)
        return 0.99 <= norm_sq <= 1.01
    
    # Test 1: NOT preserves normalization
    q.set_value([0.6, 0.8])
    q.apply_not(0)
    if not is_normalized(q.state):
        return False
    
    # Test 2: H preserves normalization
    q.set_value([0.6, 0.8])
    q.apply_h(0)
    if not is_normalized(q.state):
        return False
    
    # Test 3: Z preserves normalization
    q.set_value([0.6, 0.8])
    q.apply_z(0)
    if not is_normalized(q.state):
        return False
    
    # Test 4: Multiple gates preserve normalization
    q.set_value([inv_sqrt2, inv_sqrt2])
    q.apply_h(0)
    q.apply_z(0)
    q.apply_not(0)
    if not is_normalized(q.state):
        return False
    
    return True


def test_repeated_gate_identities() -> bool:
    q = SingleQubitSystem()
    inv_sqrt2 = 1.0 / (2**0.5)
    
    # Test 1: NOT twice returns to original
    q.set_value([0.6, 0.8])
    original = q.get_value_vector()
    q.apply_not(0)
    q.apply_not(0)
    if not compare_lists(q.state, original):
        return False
    
    # Test 2: Z twice returns to original
    q.set_value([inv_sqrt2, inv_sqrt2])
    original = q.get_value_vector()
    q.apply_z(0)
    q.apply_z(0)
    if not compare_lists(q.state, original):
        return False
    
    # Test 3: H twice returns to original
    q.set_value([0.6, 0.8])
    original = q.get_value_vector()
    q.apply_h(0)
    q.apply_h(0)
    if not compare_lists(q.state, original):
        return False
    
    # Test 4: NOT twice on |0>
    q.set_value([1.0, 0.0])
    q.apply_not(0)
    q.apply_not(0)
    if not compare_lists(q.state, [1.0, 0.0]):
        return False
    
    # Test 5: H twice on |1>
    q.set_value([0.0, 1.0])
    q.apply_h(0)
    q.apply_h(0)
    if not compare_lists(q.state, [0.0, 1.0]):
        return False
    
    return True


def test_index_validation_consistency() -> bool:
    q = SingleQubitSystem()
    
    # Test 1: All single-qubit gates reject index 1
    try:
        q.apply_not(1)
        return False
    except IndexError:
        pass
    
    try:
        q.apply_h(1)
        return False
    except IndexError:
        pass
    
    try:
        q.apply_z(1)
        return False
    except IndexError:
        pass
    
    # Test 2: All single-qubit gates reject negative indices
    try:
        q.apply_not(-1)
        return False
    except IndexError:
        pass
    
    try:
        q.apply_h(-1)
        return False
    except IndexError:
        pass
    
    try:
        q.apply_z(-1)
        return False
    except IndexError:
        pass
    
    # Test 3: All single-qubit gates accept index 0
    q.set_value([1.0, 0.0])
    try:
        q.apply_not(0)
        q.apply_h(0)
        q.apply_z(0)
    except IndexError:
        return False
    
    return True




def run_tests() -> None:
    tests = [
        ("test_set_value", test_set_value),
        ("test_get_value_vector", test_get_value_vector),
        ("test_apply_not", test_apply_not),
        ("test_apply_h", test_apply_h),
        ("test_apply_z", test_apply_z),
        ("test_apply_cnot", test_apply_cnot),
        ("test_apply_swap", test_apply_swap),
        ("test_measure", test_measure),
        ("test_measurement_idempotence", test_measurement_idempotence),
        ("test_set_value_defensive_copy", test_set_value_defensive_copy),
        ("test_gate_normalization_preservation", test_gate_normalization_preservation),
        ("test_repeated_gate_identities", test_repeated_gate_identities),
        ("test_index_validation_consistency", test_index_validation_consistency),
    ]

    for name, fn in tests:
        try:
            result = fn()
        except Exception as e:
            print(f'Exception on {name}:', e)
            result = False

        print(f"{name}: {'PASS' if result else 'FAIL'}")

run_tests()


# %% [markdown]
# ## Task 4: Implement everything else!
# Remember to be thorough with your test cases! For example, even though our `set_value` test is now non-trivial, it is far from exhaustive; it does not test for:
# 
# 
# 1.   String inputs: Per the specs SingleQubit should support the following strings representing bra-ket states: `"|0>"`, `"|1>"`, `"|+>"` and `"|->"`
# 2.   States with an incorrect number of qubits
# 3.   Invalid states
# 
# Good luck!

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
# 3.   Rename the downloaded file to `Lab4Answers.py`.
# 4.   Upload the `Lab4Answers.py` file to Gradescope.
# 5.   Ensure the autograder runs successfully.


