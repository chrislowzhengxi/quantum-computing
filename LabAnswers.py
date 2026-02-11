# %% [markdown]
# # Introduction
# In this lab you will be exploring applications of entanglement, how to create your own custom gates, analyzing a Qiskit circuit, and adapting a circuit to a device specification.
# 
# # Some helpful programming hints:
# Some helpful programming hints:
# 
# - The line circuit.draw(), where circuit is your Qiskit circuit, will draw out the circuit so you can visualize it. This must be the final call in a cell in order for the circuit to be rendered, alternatively, you can call ```print(circuit)``` at any point to see an ascii representation of the circuit
# - op = qiskit.quantum_info.Operator(circuit) will create an operator object, and op.data will let you look at the overall matrix for a circuit.
# - Keep in mind that Qiskit has a different relationship between the drawing and mathematical representation than we have. Specifically, they place the left-most bit at the bottom rather than at the top. You can [**watch this video**](https://youtu.be/Gf7XFFKS9jE) for more information. This has several implications.
# - If you look at a circuit the way we do, then the state vector ends up being stored as \[00, 10, 01, 11\] rather than \[00, 01, 10, 11\] (where the qubit on top is still the left-most qubit).
# - In reality, though, Qiskit also considers the qubit order to be swapped (little endian), where the top qubit is the least significant (right-most) bit. That is for qubits from top to bottom q0, q1, q2, the bitstring is q2, q1, q0. So the state vector is still \[00, 01, 10, 11\] from this perspective. We can see this in the CX gate.
# 
# ```
# q0_0: ──■──  
#       ┌─┴─┐  
# q0_1: ┤ X ├  
#       └───┘  
# ```
#    
# This ordering also affects the matrix, resulting in the following for CX:  
# ```
# [[1, 0, 0, 0],  
#  [0, 0, 0, 1],  
#  [0, 0, 1, 0],  
#  [0, 1, 0, 0]]  
# ```
# 
# Which will take \[00: w, 01: x, 10: y, 11: z\] to \[00: w, 01: z, 10: y, 11: x\] in little endian form, and \[00: w, 01: y, 10: z, 11: x\] in big endian form (most significant bit first).
# 
# **You are allowed to use Numpy and Networkx in addition to the python standard library**
# 
# # Grading:  
# - The output matrix of your circuit will be compared against a baseline circuit, your circuit will be compared against this matrix.
# - If they do not match, we will test the circuit with different inputs and compare against the expected values.
# - You will receive feedback for whether the circuit runs. If it does not, you will receive an error message. If it runs with no message, it means that your circuit runs, but not necessarily that the answer is correct.
# - **Do not change any function names or return types**
# 
# 

# %% [markdown]
# # Exercise 1: Teleportation
# 
# You are given a circuit with two qubits qubit_pair, represented as a tuple of two qubits, in a Bell state. The entangled pair can be in any possible Bell Pair (i.e., starting in |00>, |01>, |10>, or |11> before being entangled). The circuit also has a third qubit, outside_qubit. Write a function that transfers the state from the outside qubit to the second qubit in the Bell pair.
# 
# circuit: initialized qiskit circuit, add your gates here
# outside_qubit: the qubit whose value you will be teleporting
# qubit_pair: tuple containing the indices of the two entangled qubits
# bell_pair_start: the starting state of the qubit pair before they were put through the entanglement circuit to create a bell pair represented as a two character string: (`'00'`,`'01'`,`'10'`, or `'11'`)
# 
# 
# Specifically, you will be implementing the `teleportation circuit` portion of the following diagram:
# 
# ![](https://www.classes.cs.uchicago.edu/archive/2026/winter/22880-1/assigns/week7/teleportation_circuit.png)
# 
# The diagram assumes a `bell_pair_start` of `00`, how would you modify the circuit to handle different starting states?
# 
# 
# For ease of grading, **please do not add measurement gates to your circuit**. It is not explicitly necessary to demonstrate the transfer of state.
# 

# %%
import qiskit
from qiskit.circuit.library import UnitaryGate
import types

if not hasattr(qiskit, "extensions"):
    qiskit.extensions = types.SimpleNamespace(UnitaryGate=UnitaryGate)

def hw4_1_response(circuit, outside_qubit, qubit_pair, bell_pair_start):
    # Normalize to the standard Bell pair if needed.
    if bell_pair_start[0] == "1":
        circuit.z(qubit_pair[1])
    if bell_pair_start[1] == "1":
        circuit.x(qubit_pair[1])

    circuit.cx(outside_qubit, qubit_pair[0])
    circuit.h(outside_qubit)
    circuit.cx(qubit_pair[0], qubit_pair[1])
    circuit.cz(outside_qubit, qubit_pair[1])
    return circuit
      

# %%
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, state_fidelity

def _make_bell_pair(circuit, qubit_pair, bell_pair_start):
    if bell_pair_start[0] == "1":
        circuit.x(qubit_pair[0])
    if bell_pair_start[1] == "1":
        circuit.x(qubit_pair[1])
    circuit.h(qubit_pair[0])
    circuit.cx(qubit_pair[0], qubit_pair[1])

def _teleport_and_check(label, bell_pair_start="00"):
    qc = QuantumCircuit(3)
    outside = 0
    alice, bob = 1, 2

    if label == "1":
        qc.x(outside)
    elif label == "+":
        qc.h(outside)

    _make_bell_pair(qc, (alice, bob), bell_pair_start)
    hw4_1_response(qc, outside, (alice, bob), bell_pair_start)

    sv = Statevector.from_instruction(qc)
    rho_bob = partial_trace(sv, [outside, alice])
    expected = DensityMatrix.from_instruction(QuantumCircuit(1))
    if label == "1":
        expected = DensityMatrix.from_label("1")
    elif label == "+":
        expected = DensityMatrix.from_label("+")

    fidelity = state_fidelity(rho_bob, expected)
    print(f"label={label}, bell_pair_start={bell_pair_start}, fidelity={fidelity:.6f}")

for start in ["00", "01", "10", "11"]:
    _teleport_and_check("1", start)
    _teleport_and_check("+", start)

# %% [markdown]
# # Exercise 2: Making Gates
# Create a function that, given a list of n-bit codes and the length of the code, creates a gate that acts on n+1 qubits, and implements the Archimedes Oracle. Then add it to an n-qubit circuit, and return the circuit from the function.
# 
# 
# 
# Remember that a Qiskit uses a different ordering of states, where the top qubit is the least significant qubit when creating bitstrings.
# 
# In our convention, the top qubit is the most significant bit, and a Qiskit matrix acting on a three qubit state vector will act on the state vector as if it was \[000, 100, 010, 110, 001, 101, 011, 111\].
# 
# This means that the response bit will be 'at the top' of the circuit, if the code `001` is included then your matrix should map
# $$ |0100\rangle$$
# to
#  $$ |1100\rangle$$
# 
# There is documentation on creating your own, custom gate [**here**](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.UnitaryGate)
# 

# %%
from qiskit.circuit.library import UnitaryGate
import numpy as np

def hw4_2_response(circuit, n, codes):
    code_set = {int(code, 2) for code in codes}
    num_qubits = n + 1
    dim = 2 ** num_qubits
    unitary = np.zeros((dim, dim), dtype=complex)

    for in_index in range(dim):
        input_value = in_index >> 1
        out_index = in_index
        if input_value in code_set:
            out_index = in_index ^ 1
        unitary[out_index, in_index] = 1.0

    gate = UnitaryGate(unitary)
    circuit.append(gate, list(range(num_qubits)))
    return circuit


# %%
from qiskit.quantum_info import Operator

def _check_oracle(codes):
    qc = QuantumCircuit(4)
    hw4_2_response(qc, 3, codes)
    op = Operator(qc).data
    # Show a few basis transitions for inputs; target bit is last.
    for input_code in ["000", "001", "010", "011", "110"]:
        in_index = int(input_code, 2) << 1
        out_index = int(input_code, 2) << 1
        if input_code in codes:
            out_index ^= 1
        print(f"{input_code}0 -> {input_code}{out_index & 1}")
        # Sanity: unitary maps |in_index> to |out_index>
        col = op[:, in_index]
        print(f"  nonzero at {np.argmax(np.abs(col))}")

_check_oracle(["000", "010", "110"])

# %% [markdown]
# # Exercise 3: Analyzing Circuits
# 
# Write a function that given an n qubit circuit, returns a length n bitstring presenting the code for the Bernstein Vazarani Oracle embedded in the circuit, and the target of the oracle. Your bitstring should use an "x" to represent the location of the target, and 0s and 1s to represent the rest of the code and treat Qubit 0 as the most signficant bit.
# 
# The only CX gates included in this circuit are involved in the oracle. You should do this without simulating the circuit, only analyzing the different gates in the circuit.
# 
# You can examine the different operations in a circuit with a for loop over the circuit:
# ```python
# for gate in circuit:
#     print(gate)
# ```

# %%
def hw4_3_response(circuit):
    num_qubits = circuit.num_qubits
    targets = {}
    controls_by_target = {}

    for inst in circuit:
        op = inst.operation
        if op.name not in ("cx", "cnot"):
            continue
        control_bit = circuit.find_bit(inst.qubits[0]).index
        target_bit = circuit.find_bit(inst.qubits[1]).index

        targets[target_bit] = targets.get(target_bit, 0) + 1
        controls_by_target.setdefault(target_bit, set()).add(control_bit)

    if targets:
        target = max(targets.items(), key=lambda item: item[1])[0]
    else:
        target = num_qubits - 1
        controls_by_target[target] = set()

    bitstring = ["0"] * num_qubits
    for control in controls_by_target.get(target, set()):
        bitstring[control] = "1"
    bitstring[target] = "x"
    return "".join(bitstring)

# %%
from qiskit import QuantumCircuit

def _build_bv_oracle(code, target=None):
    n = len(code)
    qc = QuantumCircuit(n)
    if target is None:
        target = n - 1
    for i, bit in enumerate(code):
        if i == target:
            continue
        if bit == "1":
            qc.cx(i, target)
    return qc

# Target is last qubit
for test_code in ["101", "010", "111"]:
    qc = _build_bv_oracle(test_code)
    print(f"code={test_code} -> {hw4_3_response(qc)}")

# Target is middle qubit (index 1)
qc_mid = QuantumCircuit(3)
qc_mid.cx(0, 1)
qc_mid.cx(2, 1)
print(f"target=1 -> {hw4_3_response(qc_mid)}")

# %% [markdown]
# # Submission
# Congratulations on completing the lab!  
# Make sure you:
# 1. Download your lab as a python script (File-> Save and Export Notebook As...->Executable Script
# 2. Rename the downloaded file to **LabAnswers.py**
# 3. Upload the **LabAnswers.py** file to gradescope
# 4. Ensure the autograder runs successfully 


