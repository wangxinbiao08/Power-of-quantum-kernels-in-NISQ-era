import pennylane as qml
import qiskit
import qiskit.providers.aer.noise as noise
import numpy as np


# from quantum_kernel import get_args

# args = get_args()
def phi_calcu(x):
    x_shape = x.shape[0]
    phi_x = np.zeros((x_shape, x_shape))
    for i in range(x_shape):
        for j in range(x_shape):
            phi_x[i][j] = (np.pi - x[i]) * (np.pi - x[j])
    return phi_x


def my_quantum_function(x, z, phix, phiz):
    nqubit = x.shape[0]

    # encoding x data
    for i in range(nqubit):
        qml.Hadamard(wires=i)
    for i in range(nqubit):
        qml.RZ(x[i].val, wires=i)
    for i in range(nqubit - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(phix[i][i + 1], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    for i in range(nqubit):
        qml.Hadamard(wires=i)
    for i in range(nqubit):
        qml.RZ(x[i].val, wires=i)
    for i in range(nqubit - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(phix[i][i + 1], wires=i + 1)

        # encoding z data
    for i in range(nqubit - 1):
        qml.RZ(-phiz[i][i + 1], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    for i in range(nqubit):
        qml.RZ(-z[i].val, wires=i)
    for i in range(nqubit):
        qml.Hadamard(wires=i)
    for i in range(nqubit - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(-phiz[i][i + 1], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    for i in range(nqubit):
        qml.RZ(-z[i].val, wires=i)
    for i in range(nqubit):
        qml.Hadamard(wires=i)

    return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))


class DataEncoding():
    def __init__(self, args):

        SHOTS = args.SHOTS
        prob_1 = args.prob1  # 1-qubit gate
        prob_2 = args.prob2  # 2-qubit gate
        # Depolarizing quantum errors
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)
        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

        self.dev1 = qml.device('qiskit.aer', wires=2, noise_model=noise_model, shots=SHOTS)
        # self.dev2 = qml.device('default.qubit', wires=2)

        self.my_qnode = qml.QNode(my_quantum_function, self.dev1)

    def myqnode(self, a, b, c, d):
        #return qml.QNode(my_quantum_function, self.dev1)
        return self.my_qnode(a, b, c, d)

    def quantum_kernel_matrix(self, data):
        n_samples = data.shape[0]
        kernel_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                phi_x = phi_calcu(data[i, :])
                phi_z = phi_calcu(data[j, :])
                kernel_matrix[i][j] = np.mean(np.sum(self.myqnode(data[i, :], data[j, :], phi_x, phi_z), axis=0) == 2)
                kernel_matrix[j][i] = kernel_matrix[i][j]
        return kernel_matrix

    # @qml.qnode(device=self.dev1)
