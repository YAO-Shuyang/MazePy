import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.float64_t, ndim=2] _convolve2d(
    cnp.ndarray[cnp.float64_t, ndim=2] signal,
    cnp.ndarray[cnp.float64_t, ndim=1] kernel,
    int axis
):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] convolved_signal = np.zeros_like(
        signal, np.float64
    )

    if axis == 1:
        for i in range(signal.shape[0]):
            convolved_signal[i, :] = np.convolve(signal[i, :], kernel, mode='same')
    elif axis == 0:
        for i in range(signal.shape[1]):
            convolved_signal[:, i] = np.convolve(signal[:, i], kernel, mode='same')

    return convolved_signal