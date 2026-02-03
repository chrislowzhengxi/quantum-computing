# %% [markdown]
# # Introduction
# These labs will be an introduction to the Qiskit Framework, which is a Python package, developed by IBM, for construction, testing, optimizing, simulating and running quantum circuits on real quantum computers!
# 
# Qiskit works by declaratively building up quantum circuits by creating Quantum and Classical Registers, creating a circuit object and then adding gates to the circuit that act on specific qubits.
# # This Lab
# In this particular lab assignment, these exercises build up to more complex circuits, and you will be creating circuits based on a given input and a given output.
# 
# When considering the state of a circuit in this lab, consider the "top" qubit, qubit 0 in a bitstring representing the state of the circuit. The following circuit numbering is a representation of where each qubit should be considered to be in the bitstring.
# 
# # Some helpful programming hints
# 
# - The line `circuit.draw()`, where `circuit` is your Qiskit circuit, will draw out the circuit so you can visualize it.
# - `op = qiskit.quantum_info.Operator(circuit)` will create an operator object, and `op.data` will let you look at the overall matrix for a circuit.
# - Keep in mind that Qiskit has a different relationship between the drawing and mathematical representation than we have. Specifically, they place the left-most bit at the bottom rather than at the top. You can [**watch this video**](https://youtu.be/Gf7XFFKS9jE) for more information. This has several implications.
# - If you look at a circuit the way we do, then the state vector ends up being stored as `[00, 10, 01, 11]` rather than `[00, 01, 10, 11]` (where the qubit on top is still the left-most qubit).
# - In reality, though, Qiskit also considers the qubit order to be swapped (little endian), where the top qubit is the least significant (right-most) bit. That is for qubits from top to bottom `q0`, `q1`, `q2`, the bitstring is `q2`, `q1`, `q0`. So the state vector is still `[00, 01, 10, 11]` from this perspective. We can see this in the CX gate.
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
# Which will take `[00: w, 01: x, 10: y, 11: z]` to `[00: w, 01: z, 10: y, 11: x]` in little endian form, and `[00: w, 01: y, 10: z, 11: x]` in big endian form (most significant bit first).
# 
# # Grading
# - The output matrix of your circuit will be compared against a baseline circuit, your circuit will be compared against this matrix.
# - If they do not match, we will test the circuit with different inputs and compare against the expected values.
# - You will receive feedback for whether the circuit runs. If it does not, you will receive an error message. If it runs with no message, it means that your circuit runs, but not necessarily that the answer is correct.
# - **Do not change any function names or return types**.
# 
# # Qiskit Example
# This is an example for how to build a circuit using Qiskit. It will go over how to create a circuit and apply basic gates to the qubits.
# 
# # Circuit Creation
# 
# We create a quantum circuit by first creating a quantum register with the number of qubits that will be included in the overall quantum circuit. We can also create a classical register that will hold the measurement outcomes if there any measurement operations in the circuit.
# ``` python
# # Allocate a 3-Qubit Quantum Register
# qrex = qiskit.QuantumRegister(3)
# # Allocate a 3-Bit Classical Register
# crex = qiskit.ClassicalRegister(3)
# # Create a Quantum Circuit with a 3-Qubit Quantum Register and a 3-Bit Classical Register
# qcex = qiskit.QuantumCircuit(qrex, crex)
# ```
# 
# # Gate Addition
# For most of the circuits we will be using, gates are methods of the quantum circuit. For a single-qubit gate, indicate which qubit the gate is being applied to. For a two-qubit gate like the CNOT gate, the order of the qubits matters (control and target). For the two-qubit SWAP gate, the order doesn't matter.
# 
# ``` python
# qcex.x(qrex[0]) # Apply an X gate to the first qubit.
# qcex.z(qrex[1]) # Apply a Z gate to the second qubit.
# qcex.h(qrex[2]) # Apply an H gate to the third qubit.
# qcex.cx(qrex[0], qrex[1]) # Apply a CNOT gate where the control is the first qubit and the target is the second.
# qcex.cx(qrex[1], qrex[0]) # Apply a CNOT gate where the control is the second qubit and the target is the first.
# qcex.swap(qrex[1], qrex[2]) # Apply a SWAP gate between the second and third qubits.
# 
# qcex.measure(qrex, crex) # Applies a measurement across all the qubits to the classical register
# 
# # To look at the circuit diagram, use .draw()
# qcex.draw()
# ```
# 

# %% [markdown]
# # Pre-Exercise
# ```
#                 ┌───┐
# q0_0: ──■───────┤ H ├──X───────
#       ┌─┴─┐┌───┐└───┘  |  ┌───┐
# q0_1: ┤ X ├┤ X ├───────|──┤ X ├
#       └───┘└─┬─┘┌───┐  |  └───┘
# q0_2: ───────■──┤ Z ├──X───────
#                 └───┘
#     
# ```
# Recreate the following circuit in Qiskit. Don't worry about spacing, just get the gates in the right order. The number of qubits in the register needs to match the number of qubits in the pictured circuit. You may need to adjust this in the code provided.
# 

# %%
## RUN THIS CELL TO INSTALL QISKIT & OTHER RESOURCES
## (Press Shift+Enter or click on )
!pip install qiskit
!pip install qiskit_ibm_runtime
!pip install matplotlib
!pip install pylatexenc
!pip install qiskit-aer

# %%
import qiskit

def hw1_0_response():
    qrpre = qiskit.QuantumRegister(3)
    qcpre = qiskit.QuantumCircuit(qrpre)

    # Put your code here (spaces for indentation)
    qcpre.cx(qrpre[0], qrpre[1])   # black dot on q0_0, X on q0_1
    qcpre.cx(qrpre[2], qrpre[1])

    qcpre.h(qrpre[0])              # H on q0_0
    qcpre.z(qrpre[2])              # Z on q0_2

    qcpre.swap(qrpre[0], qrpre[2]) # two X's connected vertically

    qcpre.x(qrpre[1])              # final standalone X


    # End Code

    return qcpre

# %% [markdown]
# ## Testing your code
# All of your solutions will be formatted as functions. This means that your code will not be run unless you explicitly call its function. **This makes it so that the Python interpreter will not catch any errors until you run your code!**  [Read more about Python's laziness.](https://en.wikipedia.org/wiki/Lazy_evaluation)
# 
# In order to test your code you will have to call each function as follows:

# %%
hw1_0_response()

# %% [markdown]
# You might also want to save the result returned by your function and draw it with `.draw()`. Make sure to run all of your code before submitting your lab!

# %%
qc = hw1_0_response()
qc.draw()

# %%
from qiskit.quantum_info import Operator
Operator(hw1_0_response()).data


# %% [markdown]
# # Exercise 1: 1 Qubit Circuit
# Starting in state $|0\rangle$ create a circuit that generates a $\frac{1}{\sqrt{2}} (|0\rangle - |1\rangle)$ state, or $|-\rangle$.
# 
# You may include helper functions if needed.

# %%
import qiskit

def hw1_1_response():
    qr1 = qiskit.QuantumRegister(1)
    qc1 = qiskit.QuantumCircuit(qr1)

    # Put your code here (spaces for indentation)

    qc1.h(qr1[0])
    qc1.z(qr1[0])

    # End Code

    return qc1

# %%
qc = hw1_1_response()
qc.draw()


# %% [markdown]
# # Exercise 2: 2 Qubit Circuit
# This time, we will not assume a known starting state. Starting in unknown state $\alpha|00\rangle + \beta|11\rangle$, create a circuit that transforms that state to the $\frac{1}{\sqrt{2}} (\beta|00\rangle - \alpha|01\rangle + \beta|10\rangle + \alpha|11\rangle)$ state.
# 
# You may include helper functions if needed.

# %%
import qiskit


def hw1_2_response():


    qr2 = qiskit.QuantumRegister(2)
    qc2 = qiskit.QuantumCircuit(qr2)


    # Put your code here (spaces for indentation)
    # qc2.x(qr2[0])
    
    # qc2.x(qr2[1])
    
    # qc2.h(qr2[1])
    
    # qc2.x(qr2[1])
    qc2.x(qr2[1])
    
    qc2.x(qr2[0])
    
    qc2.h(qr2[0])
    
    qc2.x(qr2[0])


    return qc2

# def hw1_2_response():


#     qr2 = qiskit.QuantumRegister(2)
#     qc2 = qiskit.QuantumCircuit(qr2)


#     # Put your code here (spaces for indentation)
#     qc2.x(qr2[0])
    
#     qc2.h(qr2[1])
    
#     qc2.z(qr2[1])
    
#     qc2.x(qr2[1])


#     return qc2

# %%
hw1_2_response()

# %% [markdown]
# # Exercise 3: 2 Qubit Circuit
# Create a circuit that performs the following transformations:
# 
# - starting in state $|00\rangle$ generates a $\frac{1}{\sqrt{2}} (-|10\rangle+|11\rangle)$ state
# - starting in state $|11\rangle$ generates a $\frac{1}{\sqrt{2}}  (|00\rangle-|01\rangle)$ state.
# 
# **Hint:** You will need to use a CNOT Gate.
# 
# You may include helper functions if needed.

# %%
import qiskit

def hw1_3_response():
    qr3 = qiskit.QuantumRegister(2)
    qc3 = qiskit.QuantumCircuit(qr3)

    # Put your code here (spaces for indentation)
    # qc3.x(qr3[0])
    # qc3.cx(qr3[1], qr3[0])

    # qc3.h(qr3[0])

    # qc3.x(qr3[1])

    # qc3.z(qr3[1])

    qc3.x(qr3[1])
    qc3.cx(qr3[0], qr3[1])

    qc3.h(qr3[1])

    qc3.x(qr3[0])

    qc3.z(qr3[0])

    # End Code

    return qc3

# %%
hw1_3_response()

# %% [markdown]
# # Exercise 4: 2 Qubit Circuit
# Create a circuit that:
# 
# - starting in state $|00\rangle$ transforms to a $\frac{1}{2} (|00\rangle - |01\rangle - |10\rangle + |11\rangle)$ state,
# - starting in state $|10\rangle$ transforms to a $\frac{1}{2} (|00\rangle + |01\rangle - |10\rangle - |11\rangle)$ state,
# - starting in state $|01\rangle$ transforms to a $\frac{1}{2} (|00\rangle - |01\rangle + |10\rangle - |11\rangle)$ state
# - starting in state $|11\rangle$ transforms to a $\frac{1}{2} (|00\rangle + |01\rangle + |10\rangle + |11\rangle)$ state
# 
# 
# **Hint:** It may be helpful to consider using a SWAP gate.
# 
# You may include helper functions if needed.

# %%
import qiskit

def hw1_4_response():
    qr4 = qiskit.QuantumRegister(2)
    qc4 = qiskit.QuantumCircuit(qr4)

    qc4.x(qr4[0])
    qc4.x(qr4[1])
    
    qc4.swap(qr4[0], qr4[1])
    
    qc4.h(qr4[0])
    qc4.h(qr4[1])

    return qc4

# %%
hw1_4_response()

# %% [markdown]
# # Submission
# Congratulations on completing the lab! Make sure you:
# 1. Test all of your functions by calling them at least once.
# 2. Download your lab as a **Python** `.py` script (*not* an `.ipynb` file):
#   
#     ```File-> Download -> Download .py```
# 
# 3. Rename the downloaded file to **Lab3Answers.py**.
# 4. Upload the **Lab3Answers.py** file to Gradescope.
# 5. Ensure the autograder runs successfully.

# %% [markdown]
# 


