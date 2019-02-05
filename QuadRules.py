import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Integrand
import AnimationTools
import GMatrix
import XGrid


def TrapUnequal(G, phat, kvect):
    first = np.matmul(G[:, :-1], phat[:-1] * kvect)
    second = np.matmul(G[:, 1:], phat[1:] * kvect)
    half = (first + second) / 2
    return half
