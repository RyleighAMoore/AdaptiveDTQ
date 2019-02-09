import numpy as np


def TrapUnequal(G, phat, kvect):
    first = np.matmul(G[:, :-1], phat[:-1] * kvect)
    second = np.matmul(G[:, 1:], phat[1:] * kvect)
    half = (first + second) * 0.5
    return half
