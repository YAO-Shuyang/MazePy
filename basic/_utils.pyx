import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.int64_t, ndim=2] _convert_kilosort_spike_train(
    cnp.ndarray[cnp.int64_t, ndim=1] activity
):
    cdef:
        int n_neuron = np.max(activity)
        cnp.ndarray[cnp.int64_t, ndim=2] spike_train = np.zeros(
            (n_neuron, activity.shape[0]),
            np.int64
        )
    
    for i in range(activity.shape[0]):
        spike_train[activity[i] - 1, i] = 1

    return spike_train

