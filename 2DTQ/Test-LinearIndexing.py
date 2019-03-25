import numpy as np

A = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

A_rav = np.ravel(A)

A_unrav = np.unravel_index(A_rav, (3, 3))

arr = np.array([[0, 0, 0], [0, 1, 2], [0, 2, 3], [0, 1, 1]])
W = np.ravel_multi_index(arr, (10, 10, 10, 10))

id_1d = np.arange(9)

# getting the indices of the multi-dimensional array
idx, idy, idz = np.unravel_index(id_1d, (3, 3, 3))

# getting the 1d indecis
np.ravel_multi_index((idx, idy, idz), (3, 3, 3))

t = 0
