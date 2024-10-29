import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.int64_t, ndim=2] _get_kilosort_spike_counts(
    cnp.ndarray[cnp.int64_t, ndim=1] spikes,
    cnp.ndarray[cnp.int64_t, ndim=1] variable,
    int nbins,
    int n_neurons
):
    cdef:
        cnp.ndarray[cnp.int64_t, ndim=2] spike_counts = np.zeros(
            (n_neurons, nbins), np.int64
        )

    for i in range(spikes.shape[0]):
        spike_counts[spikes[i] - 1, variable[i]] += 1
    
    return spike_counts

cpdef cnp.ndarray[cnp.int64_t, ndim=1] _get_kilosort_spike_counts_total(
    cnp.ndarray[cnp.int64_t, ndim=1] spikes,
    int n_neurons
):
    cdef cnp.ndarray[cnp.int64_t, ndim=1] spike_counts = np.zeros(
        n_neurons, dtype=np.int64
    )

    for i in range(spikes.shape[0]):
        spike_counts[spikes[i] - 1] += 1

    return spike_counts

cpdef cnp.ndarray[cnp.int64_t, ndim=2] _get_spike_counts(
    cnp.ndarray[cnp.int64_t, ndim=2] spikes,
    cnp.ndarray[cnp.int64_t, ndim=1] variable,
    int nbins
):
    cdef:
        cnp.ndarray[cnp.int64_t, ndim=2] spike_counts = np.zeros(
            (spikes.shape[0], nbins), np.int64
        )

    for i in range(nbins): 
        idx = np.where(variable == i)[0]
        spike_counts[:, i] += np.sum(spikes[:, idx], axis = 1)
    
    return spike_counts

cpdef cnp.ndarray[cnp.int64_t, ndim=1] _get_spike_counts_total(
    cnp.ndarray[cnp.int64_t, ndim=2] spikes
):
    return np.sum(spikes, axis = 1)

cpdef cnp.ndarray[cnp.float64_t, ndim=2] _get_occu_time(
    cnp.ndarray[cnp.int64_t, ndim=1] variable,
    cnp.ndarray[cnp.float64_t, ndim=1] dtime,
    int nbins
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] occu_time = np.zeros(
        nbins,
        np.float64
    )

    for i in range(variable.shape[0]):
        occu_time[variable[i]] += dtime[i]

    return occu_time

cpdef cnp.ndarray[cnp.float64_t, ndim=2] calc_neural_trajectory(
    cnp.ndarray[cnp.int64_t, ndim = 2] spikes,
    cnp.ndarray[cnp.float64_t, ndim = 1] time,
    double t_window,
    double step_size
):
    cdef:
        double t_min = np.min(time)
        double t_max = np.max(time)
        # Determine the number of time bins (nstep+1)
        int nstep = int((t_max - t_min - t_window) // step_size) + 1
        cnp.ndarray[cnp.float64_t, ndim=2] neural_traj = np.zeros(
            (spikes.shape[0], nstep+1), np.float64
        )

        int left, right

        cnp.ndarray[cnp.float64_t, ndim=1] left_bounds = np.linspace(
            t_min, t_min + step_size * nstep, nstep+1
        )

        cnp.ndarray[cnp.float64_t, ndim=1] right_bounds = left_bounds + t_window

        int init_frame = 0
        int is_first = 0

    # Determine range first
    
    for i in range(nstep+1):
        # Determine the range
        for j in range(init_frame, time.shape[0]):
            if time[j] < left_bounds[i]:
                continue
            
            if is_first == 0:
                left = j
                init_frame = j
                is_first = 1

            if time[j] >= right_bounds[i]:
                right = j
                is_first = 0
                break

        # Convert to firing rate
        neural_traj[:, i] = np.sum(spikes[:, left:right], axis = 1) / t_window * 1000

    return neural_traj


cpdef cnp.ndarray[cnp.int64_t, ndim=2] _convert_to_kilosort_form(
    cnp.ndarray[cnp.int64_t, ndim=2] spikes
):
    idx_time, idx_neuron = np.where(spikes.T == 1)

    return np.vstack((idx_neuron+1, idx_time))