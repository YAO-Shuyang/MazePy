# time_sync.pyx
import numpy as np
cimport numpy as cnp

def coordinate_recording_time(
    cnp.ndarray[cnp.float64_t, ndim=1] source_time, 
    cnp.ndarray[cnp.float64_t, ndim=1] target_time
) -> cnp.ndarray:
    """
    Coordinate recording time of behavioral data and neural data.

    Parameters
    ----------
    source_time : ndarray
        The time to be converted (typically the neural data).
    target_time : ndarray
        The targeted time (typically the behavioral data).

    Returns
    -------
    ndarray
        The index of each source_time in target_time. The same length as
        source_time.

    Examples
    --------
    >>> source_time = np.array([0, 33, 67, 100, 133, 167, 200, 233, 267, 300])
    >>> target_time = np.array([0, 50, 100, 150, 200, 250, 300])
    >>> coordinate_recording_time(source_time, target_time)
    array([0, 1, 1, 2, 3, 3, 4, 5, 5, 6], dtype=int64)
    """
    cdef int i, j, min_idx
    cdef double min_val
    cdef cnp.ndarray[cnp.int64_t, ndim=1] res = np.zeros(source_time.shape[0], dtype=np.int64)
    
    for i in range(source_time.shape[0]):
        min_val = float('inf')
        min_idx = 0
        for j in range(target_time.shape[0]):
            if abs(target_time[j] - source_time[i]) < min_val:
                min_val = abs(target_time[j] - source_time[i])
                min_idx = j
        res[i] = min_idx
    return res
