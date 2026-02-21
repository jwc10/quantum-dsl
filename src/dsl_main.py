from .pl_executor import run_source_with_pennylane

source = """
qubit(q0, q1, q2, q3)
hadamard_transform(q0, q1, q2, q3)
python { combos = [[0, 1, 1, 0], [1, 1, 1, 1]]
index1 = np.ravel_multi_index(combos[0], [2]*len(combos[0])) # Index of solution 1
index2 = np.ravel_multi_index(combos[1], [2]*len(combos[1])) # Index of solution 2
my_array = np.identity(2**len(combos[0])) # Create the identity matrix
my_array[index1, index1] = -1
my_array[index2, index2] = -1
oracle = my_array
qml.QubitUnitary(oracle, wires=['q0', 'q1', 'q2', 'q3'])}
hadamard_transform(q0, q1, q2, q3)
measure_probs(q0, q1, q2, q3)
"""

result = run_source_with_pennylane(source)
print(result)
