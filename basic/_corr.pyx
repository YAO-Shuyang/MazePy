import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

# Define the type of input numpy arrays
ctypedef cnp.float64_t DTYPE_t

cpdef tuple pearsonr(cnp.ndarray[DTYPE_t, ndim=1] x, cnp.ndarray[DTYPE_t, ndim=1] y):
    cdef int n = x.shape[0]
    cdef DTYPE_t mean_x = np.mean(x)
    cdef DTYPE_t mean_y = np.mean(y)
    cdef DTYPE_t cov_xy = 0.0
    cdef DTYPE_t var_x = 0.0
    cdef DTYPE_t var_y = 0.0
    cdef int i

    # Calculate covariance and variances
    for i in range(n):
        cov_xy += (x[i] - mean_x) * (y[i] - mean_y)
        var_x += (x[i] - mean_x) ** 2
        var_y += (y[i] - mean_y) ** 2

    # Avoid division by zero
    if n > 1:
        cov_xy /= n
        var_x /= n
        var_y /= n
    else:
        return 0.0, cov_xy  # Not enough data points

    # Handle the case where variance is zero
    if var_x == 0 or var_y == 0:
        return 0.0, cov_xy  # Return correlation of zero if any variance is zero

    correlation = cov_xy / (sqrt(var_x) * sqrt(var_y))
    return correlation, cov_xy

cpdef cnp.ndarray[DTYPE_t, ndim=2] pearsonr_pairwise(cnp.ndarray[DTYPE_t, ndim=2] data, int axis=0):
    cdef int n, m, i, j
    n, m = data.shape[0], data.shape[1]
    cdef cnp.ndarray[DTYPE_t, ndim=2] corr_matrix = np.empty((m, m), dtype=np.float64) if axis == 0 else np.empty((n, n), dtype=np.float64)

    if axis == 0:  # column-wise
        for i in range(m):
            for j in range(i, m):
                corr_matrix[i, j], _ = pearsonr(data[:, i], data[:, j])
                corr_matrix[j, i] = corr_matrix[i, j]
    else:  # row-wise
        for i in range(n):
            for j in range(i, n):
                corr_matrix[i, j], _ = pearsonr(data[i, :], data[j, :])
                corr_matrix[j, i] = corr_matrix[i, j]

    return corr_matrix
