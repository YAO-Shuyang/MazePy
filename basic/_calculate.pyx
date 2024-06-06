import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.float64_t, ndim=1] _calc_information(
    cnp.ndarray[cnp.float64_t, ndim=2] firing_rate,
    cnp.ndarray[cnp.float64_t, ndim=1] mean_rate,
    cnp.ndarray[cnp.float64_t, ndim=1] occu_time_norm
):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] logArg = (firing_rate.T / mean_rate).T
    logArg[logArg == 0] = 1
    
    return np.nansum(occu_time_norm * logArg  * np.log2(logArg), axis=1)
