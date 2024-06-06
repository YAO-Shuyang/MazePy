import numpy as np
cimport numpy as cnp

from mazepy.basic._time_sync import _coordinate_recording_time
from mazepy.basic._calc_rate import _get_kilosort_spike_counts
from mazepy.basic._calc_rate import _get_kilosort_spike_counts_total
from mazepy.basic._calc_rate import _get_spike_counts, _get_occu_time
from mazepy.basic._calc_rate import _get_spike_counts_total
from mazepy.basic._calc_rate import _convert_to_kilosort_form
from mazepy.basic._calculate import _calc_information

cpdef cnp.ndarray[cnp.float64_t, ndim=2] _shift_shuffle_kilosort(
    cnp.ndarray[cnp.int64_t, ndim=1] spikes,
    cnp.ndarray[cnp.float64_t, ndim=1] dtime,
    cnp.ndarray[cnp.int64_t, ndim = 1] variable,
    int nbins,
    int n_shuffle = 1000,
    int info_thre = 95
):
    cdef:
        int n_neuron = np.max(spikes)
        # Results: n_neuron x 3, the three columns represent
        # 1. is_passed_shuffle (0. or 1.) 2. information 3. shuffle threshold (info.)
        # Units of information: bits per spikes.
        cnp.ndarray[cnp.float64_t, ndim=2] res = np.zeros(
            (n_neuron, 3), np.float64
        )
        
        cnp.ndarray[cnp.float64_t, ndim=1] occu_time = _get_occu_time(
            variable=variable, dtime=dtime, nbins=nbins
        )
        double t_total = np.nansum(occu_time)
        cnp.ndarray[cnp.float64_t, ndim=1] occu_time_norm = occu_time / t_total
        cnp.ndarray[cnp.float64_t, ndim=2] tuning_curve_shuf = np.zeros(
            (n_neuron, nbins)
        )
        cnp.ndarray[cnp.float64_t, ndim=2] tuning_curve = np.zeros(
            (n_neuron, nbins)
        )

        cnp.ndarray[cnp.int64_t, ndim=2] shuf_spikes = np.zeros(
            (n_shuffle, spikes.shape[0]), np.int64
        )

        cnp.ndarray[cnp.float64_t, ndim=2] shuf_info = np.zeros(
            (n_neuron, n_shuffle), np.float64
        )
    
        cnp.ndarray[cnp.float64_t, ndim=1] mean_rate = np.zeros(n_neuron)

        cnp.ndarray[cnp.int64_t, ndim=1] shuf_shift = np.random.randint(
            low=1, high=spikes.shape[1], size=n_shuffle
        ).astype(np.int64)
    
    mean_rate = _get_kilosort_spike_counts_total(spikes) / t_total    
    tuning_curve = _get_kilosort_spike_counts(
        spikes = spikes,
        variable = variable,
        nbins = nbins
    ) / occu_time

    res[:, 1] = _calc_information(
        tuning_curve,
        mean_rate,
        occu_time_norm
    )

    for i in range(n_shuffle):

        tuning_curve_shuf = _get_kilosort_spike_counts(
            spikes = np.roll(spikes, shift = shuf_shift[i]),
            variable = variable,
            nbins = nbins
        ) / occu_time

        shuf_info[:, i] = _calc_information(
            tuning_curve_shuf,
            mean_rate,
            occu_time_norm
        )

    res[:, 2] = np.percentile(shuf_info, info_thre, axis=1)
    res[:, 0] = np.where(res[:, 1] - res[:, 2] > 0, 1, 0)
    return res

cpdef cnp.ndarray[cnp.float64_t, ndim=2] _shift_shuffle(
    cnp.ndarray[cnp.int64_t, ndim=2] spikes,
    cnp.ndarray[cnp.float64_t, ndim=1] dtime,
    cnp.ndarray[cnp.int64_t, ndim = 1] variable,
    int nbins,
    int n_shuffle = 1000,
    int info_thre = 95
):
    # Convert spikes to kilosort_form for fast shuffle test.
    cdef cnp.ndarray[cnp.int64_t, ndim=2] res = _convert_to_kilosort_form(
        spikes
    )

    return _shift_shuffle_kilosort(
        spikes = res[0, :],
        dtime = dtime[res[1, :]],
        variable = variable[res[1, :]],
        nbins = nbins,
        n_shuffle = n_shuffle,
        info_thre = info_thre
    )

cpdef cnp.ndarray[cnp.float64_t, ndim=2] _isi_shuffle_kilosort(
    cnp.ndarray[cnp.int64_t, ndim=1] spikes,
    cnp.ndarray[cnp.float64_t, ndim=1] dtime,
    cnp.ndarray[cnp.int64_t, ndim = 1] variable,
    int nbins,
    int n_shuffle = 1000,
    int info_thre = 95
):
    cdef:
        int n_neuron = np.max(spikes)
        # Results: n_neuron x 3, the three columns represent
        # 1. is_passed_shuffle (0. or 1.) 2. information 3. shuffle threshold (info.)
        # Units of information: bits per spikes.
        cnp.ndarray[cnp.float64_t, ndim=2] res = np.zeros(
            (n_neuron, 3), np.float64
        )
        
        cnp.ndarray[cnp.float64_t, ndim=1] occu_time = _get_occu_time(
            variable=variable, dtime=dtime, nbins=nbins
        )
        double t_total = np.nansum(occu_time)
        cnp.ndarray[cnp.float64_t, ndim=1] occu_time_norm = occu_time / t_total
        cnp.ndarray[cnp.float64_t, ndim=2] tuning_curve_shuf = np.zeros(
            (n_neuron, nbins)
        )
        cnp.ndarray[cnp.float64_t, ndim=2] tuning_curve = np.zeros(
            (n_neuron, nbins)
        )

        cnp.ndarray[cnp.int64_t, ndim=2] shuf_spikes = np.zeros(
            (n_shuffle, spikes.shape[0]), np.int64
        )

        cnp.ndarray[cnp.float64_t, ndim=2] shuf_info = np.zeros(
            (n_neuron, n_shuffle), np.float64
        )
    
        cnp.ndarray[cnp.float64_t, ndim=1] mean_rate = np.zeros(n_neuron)

        cnp.ndarray[cnp.int64_t, ndim=1] shuf_shift = np.random.randint(
            low=1, high=spikes.shape[1], size=n_shuffle
        ).astype(np.int64)
    
    mean_rate = _get_kilosort_spike_counts_total(spikes) / t_total    
    tuning_curve = _get_kilosort_spike_counts(
        spikes = spikes,
        variable = variable,
        nbins = nbins
    ) / occu_time

    res[:, 1] = _calc_information(
        tuning_curve,
        mean_rate,
        occu_time_norm
    )

    for i in range(n_shuffle):

        tuning_curve_shuf = _get_kilosort_spike_counts(
            spikes = np.roll(spikes, shift = shuf_shift[i]),
            variable = variable,
            nbins = nbins
        ) / occu_time

        shuf_info[:, i] = _calc_information(
            tuning_curve_shuf,
            mean_rate,
            occu_time_norm
        )

    res[:, 2] = np.percentile(shuf_info, info_thre, axis=1)
    res[:, 0] = np.where(res[:, 1] - res[:, 2] >= 0, 1, 0)
    return res

cpdef cnp.ndarray[cnp.float64_t, ndim=2] _isi_shuffle(
    cnp.ndarray[cnp.int64_t, ndim=2] spikes,
    cnp.ndarray[cnp.float64_t, ndim=1] dtime,
    cnp.ndarray[cnp.int64_t, ndim = 1] variable,
    int nbins,
    int n_shuffle = 1000,
    int info_thre = 95
):
    cdef:
        # Results: n_neuron x 3, the three columns represent
        # 1. is_passed_shuffle (0. or 1.) 2. information 3. shuffle threshold (info.)
        # Units of information: bits per spikes.
        cnp.ndarray[cnp.float64_t, ndim=2] res = np.zeros(
            (spikes.shape[0], 3), np.float64
        )

        cnp.ndarray[cnp.float64_t, ndim=1] shuf_info = np.zeros(n_shuffle)
        
        cnp.ndarray[cnp.float64_t, ndim=1] occu_time = _get_occu_time(
            variable=variable, dtime=dtime, nbins=nbins
        )
        double t_total = np.nansum(occu_time)
        cnp.ndarray[cnp.float64_t, ndim=1] occu_time_norm = occu_time / t_total
        cnp.ndarray[cnp.float64_t, ndim=2] tuning_curve_shuf = np.zeros(
            (n_shuffle, nbins)
        )
        cnp.ndarray[cnp.float64_t, ndim=2] tuning_curve = np.zeros(
            (spikes.shape[0], nbins)
        )

        cnp.ndarray[cnp.int64_t, ndim=2] shuf_shift = np.random.randint(
            low=1, 
            high=spikes.shape[0], 
            size=(spikes.shape[0], n_shuffle)
        ).astype(np.int64)

        cnp.ndarray[cnp.int64_t, ndim=2] shuf_spikes = np.zeros(
            (n_shuffle, spikes.shape[1]), np.int64
        )
    
        cnp.ndarray[cnp.float64_t, ndim=1] mean_rate = np.zeros(spikes.shape[0])
    

    mean_rate = _get_spike_counts_total(spikes) / t_total    
    tuning_curve = _get_spike_counts(
        spikes = spikes,
        variable = variable,
        nbins = nbins
    ) / occu_time

    res[:, 1] = _calc_information(
        tuning_curve,
        mean_rate,
        occu_time_norm
    )

    for n in range(spikes.shape[0]):
        if n % 10 == 0:
            print(n)

        for i in range(n_shuffle):
            shuf_spikes[i, shuf_shift[n, i]:] = spikes[n, :spikes.shape[1]-shuf_shift[n, i]]
            shuf_spikes[i, :shuf_shift[n, i]] = spikes[n, spikes.shape[1]-shuf_shift[n, i]:]

        tuning_curve_shuf = _get_spike_counts(
            spikes = shuf_spikes,
            variable = variable,
            nbins = nbins
        ) / occu_time

        shuf_info = _calc_information(
            tuning_curve_shuf,
            np.repeat(mean_rate[n], n_shuffle),
            occu_time_norm
        )

        res[n, 2] = np.percentile(shuf_info, info_thre)

    res[:, 0] = np.where(res[:, 1] - res[:, 2] >= 0, 1, 0)
    return res