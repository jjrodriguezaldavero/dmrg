import numpy as np

sX = np.array([[0, 1], [1, 0]], dtype=float)
sZ = np.array([[1, 0], [0, -1]], dtype=float)
htemp = -np.kron(sX, sX) - 0.5 * (
    np.kron(sZ, np.eye(2)) + np.kron(np.eye(2), sZ))
hbig = (0.5 * np.kron(np.eye(4), htemp) +
        np.kron(np.eye(2), np.kron(htemp, np.eye(2))) +
        0.5 * np.kron(htemp, np.eye(4))).reshape(2, 2, 2, 2, 2, 2, 2, 2)

print(htemp)