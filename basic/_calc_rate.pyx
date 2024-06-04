import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.int64_t, ndim=2] _get_kilosort_spike_counts(
    cnp.ndarray[cnp.int64_t, ndim=1] spikes,
    cnp.ndarray[cnp.int64_t, ndim=1] variable,
    int nbins
):
    cdef:
        cnp.ndarray[cnp.int64_t, ndim=2] spike_counts = np.zeros(
            (np.max(spikes), nbins), np.int64
        )

    for i in range(spikes.shape[0]):
        spike_counts[spikes[i] - 1, variable[i]] += 1
    
    return spike_counts

cpdef cnp.ndarray[cnp.int64_t, ndim=2] _get_occu_time(
    cnp.ndarray[cnp.int64_t, ndim=1] variable,
    cnp.ndarray[cnp.float64_t, ndim=1] time,
    int nbins
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] occu_time = np.zeros(
        nbins,
        np.float64
    )

    for i in range(variable.shape[0]):
        occu_time[variable[i]] += time[i]

    return occu_time

