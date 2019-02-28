import numpy as np
import matplotlib.pyplot as plt
import GMatrix as GMatrix

def TrapUnequal(G, phat, kvect):
    first = np.matmul(G[:, :-1], phat[:-1] * kvect)
    second = np.matmul(G[:, 1:], phat[1:] * kvect)
    half = (first + second) * 0.5
    return half


def Unequal_Gk(G, kvect, xvec, h):

    GA = np.zeros((len(kvect)+1,len(kvect)+1))

    for col in range(len(G)):  # interiors
        for row in range(1,len(G)-1):
            #GA[row,col] = ((G[row, col]*(xvec[col]-xvec[col-1])) + (G[row, col]*(xvec[col+1]-xvec[col])))*0.5
            GA[row,col] = ((G[row, col]*(xvec[row]-xvec[row-1])) + (G[row, col]*(xvec[row+1]-xvec[row])))*0.5

    for col in range(len(G)):  # interiors
        #GA[row,0]= (G[row, 0])*kvect[0]
        GA[0,col]= (G[0, col])*kvect[0]*0.5

    for col in range(len(G)):  # interiors
        #GA[row,-1]= (G[row, -1])*kvect[-1]
        GA[-1,col]= (G[-1, col])*kvect[-1]*0.5


    R = np.sum(GA, axis=0)
    R2 = np.sum(GA, axis=1)

    vals, vects = np.linalg.eig(GA)
    vals = np.real(vals)
    largest_eigenvector_unscaled = vects[:, 0]
    vals2, vects2 = np.linalg.eig(G)
    vals = np.real(vals)
    largest_eigenvector_unscaledG = vects2[:, 0]
    plt.figure()

    #plt.plot(vects[:,1], label = 'GA1')
    #plt.plot(vects[:,2], label = 'GA2')
    plt.plot(largest_eigenvector_unscaledG,label = 'G')
    plt.plot(largest_eigenvector_unscaled, label='GA0')
    plt.legend()
    plt.show()
    return GA