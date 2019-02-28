import numpy as np
import matplotlib.pyplot as plt
import GMatrix as GMatrix


def TrapUnequal(G, phat, kvect):
    first = np.matmul(G[:, :-1], phat[:-1] * kvect)
    second = np.matmul(G[:, 1:], phat[1:] * kvect)
    half = (first + second) * 0.5
    return half


def Unequal_Gk(G, kvect, xvec, h):
    GA = np.zeros((len(kvect) + 1, len(kvect) + 1))

    for col in range(len(G)):  # interiors
        for row in range(1, len(G) - 1):
            GA[row, col] = ((G[row, col] * (xvec[row] - xvec[row - 1])) + (
                        G[row, col] * (xvec[row + 1] - xvec[row]))) * 0.5

    for col in range(len(G)):  # interiors
        GA[0, col] = (G[0, col]) * kvect[0] * 0.5

    for col in range(len(G)):  # interiors
        GA[-1, col] = (G[-1, col]) * kvect[-1] * 0.5

    colSums = np.sum(GA, axis=0)
    vals, vects = np.linalg.eig(GA)
    vals = np.real(vals)
    largest_eigenvector_unscaled = vects[:, 0]
    vals = np.real(vals)
    # scaled_eigvect = GMatrix.scaleEigenvector(largest_eigenvector_unscaled,kvect)
    plt.figure()
    plt.plot(xvec, largest_eigenvector_unscaled)
    plt.show()
    return GA
