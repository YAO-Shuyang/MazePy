# time_sync.pyx
import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.int64_t, ndim=1] _coordinate_recording_time(
    cnp.ndarray[cnp.float64_t, ndim=1] source_time, 
    cnp.ndarray[cnp.float64_t, ndim=1] target_time
):
    cdef int i, j, min_idx 
    cdef double min_val
    cdef cnp.ndarray[cnp.int64_t, ndim=1] res = np.zeros(
        source_time.shape[0], 
        dtype=np.int64
    )
    
    cdef int prev_index = 0
    for i in range(source_time.shape[0]):
        min_val = float('inf')
        min_idx = prev_index
        for j in range(prev_index, target_time.shape[0]):
            if abs(target_time[j] - source_time[i]) <= min_val:
                min_val = abs(target_time[j] - source_time[i])
                min_idx = j
            else:
                break
        res[i] = min_idx
        prev_index = min_idx
    return res
