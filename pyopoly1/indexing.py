import numpy as np
from scipy.special import comb


def hyperbolic_cross_indices(d, k):
    """
    Returns indices associated with a d-dimensional (isotropic)
    hyperbolic cross index space up to degree k.
    """

    from itertools import combinations
    from scipy.special import comb

    assert k >= 0
    assert d >= 1

    if d == 1:
        lambdas = range(k+1)
        return lambdas

    lambdas = np.zeros([1, d], dtype=int)

    # First add all indices with sparsity 1
    for q in range(d):
        temp = np.zeros([k-1, d], dtype=int)
        temp[:,q] = np.arange(1, k, dtype=int)
        lambdas = np.vstack([lambdas, temp])

    # Now determine the maximum 0-norm the entries can be. I.e., for
    # which values of p is 2^p <= k+1?
    pmax = int(np.floor(np.log(k+1)/np.log(2)))

    # For each sparsity p, populate with all possible indices of that
    # sparsity
    for p in range(2, pmax+1):
        # Determine all possible locations where nonzero entries can occur
        combs = combinations(range(d), p)
        combs = np.array( [row for row in combs], dtype=int)

        # Now we have 2^p < k+1, i.e., an index with nonzero entries
        # np.ones([p 1]) is ok.
        # Keep incrementing these entries until product exceeds k+1
        possible_indices = np.ones([1, p]);
        ind = 0;

        while ind < possible_indices.shape[0]:
            # Add any possibilities that are children of
            # possible_indices[ind,:]

            lambd = possible_indices[ind,:]
            for q in range(p):
                temp = lambd.copy()
                temp[q] += 1
                if np.prod(temp+1) <= k+1:
                    possible_indices = np.vstack([possible_indices, temp])

            ind += 1

        possible_indices = np.vstack({tuple(row) for row in possible_indices})
        arow = lambdas.shape[0]
        lambdas = np.vstack([lambdas, np.zeros([combs.shape[0]*possible_indices.shape[0], d], dtype=int)])

  # Now for each combination, we put in possible_indices
        for c in range(combs.shape[0]):
            i1 = arow
            i2 = arow + possible_indices.shape[0]

            lambdas[i1:i2,combs[c,:]] = possible_indices;

            arow = i2

    return lambdas


def total_degree_indices(d, k):
    # Returns multi-indices associated with d-variate polynomials of
    # degree less than or equal to k. Each row is a multi-index, ordered
    # in total-degree-graded reverse lexicographic ordering.

    assert d > 0
    assert k >= 0

    if d == 1:
        return np.arange(k+1, dtype=int).reshape([k+1, 1])

    # total degree indices up to degree k in d-1 dimensions:
    lambdasd1 = total_degree_indices(d-1, k)
    # lambdasd1 should already be sorted by total degree, which is
    # assumed below

    lambdas = np.zeros([np.round(int(comb(d+k, d))), d], dtype=int)

    i0 = 0
    for qk in range(0, k+1):

        n = int(np.round(comb(d-1+(k-qk), d-1)))
        i1 = i0 + n

        lambdas[i0:i1,0] = qk
        lambdas[i0:i1,1:] = lambdasd1[:n,:]
        i0 = i1

    # My version of numpy < 1.12, so I don't have np.flip :(
    #degrees = np.cumsum(np.flip(lambdas,axis=1), axis=1)
    degrees = np.cumsum(np.fliplr(lambdas), axis=1)

    ind = np.lexsort(degrees.transpose())
    lambdas = lambdas[ind,:]
    return lambdas

def degree_encompassing_N(d, N):
    # Returns the smallest degree k such that nchoosek(d+k,d) >= N

    k = 0
    while np.round(comb(d+k,d)) < N:
        k += 1

    return k

def total_degree_indices_N(d, N):
    # Returns the first N ( > 0) d-dimensional multi-indices when ordered by
    # total degree graded reverse lexicographic ordering.

    assert N > 0

    return total_degree_indices(d, degree_encompassing_N(d,N))[:N,:]

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    d = 1
    k = 5

    L1 = total_degree_indices(d, k)

    d = 2
    k = 7

    L2 = total_degree_indices(d, k)

    d = 4
    k = 6

    L4 = total_degree_indices(d,k)

    N = L4.shape[0] - 10
    L42 = total_degree_indices_N(d,N)

    err = np.linalg.norm( L42 - L4[:N,:])

    ############## Hyperbolic cross
    d, k = 2, 33
    lambdas = hyperbolic_cross_indices(d, k)

    plt.plot(lambdas[:,0], lambdas[:,1], 'r.')
    plt.show()
