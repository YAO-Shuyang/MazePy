"""
Provides functions for calculation.
"""

import numpy as np
from typing import Optional, Union
from mazepy.datastruc import SpikeTrain, TuningCurve, KilosortSpikeTrain

def calc_information(
    mean_rate: np.ndarray,
    rate_map: Union[TuningCurve, np.ndarray], 
    occu_time: np.ndarray
) -> float:
    """ 
    Calculate the information of neurons.
    
    Function
    --------
        ratio(i) = r_i / r_mean,\\
        lambda(i) = time at bin i / total time,\\
        Information = sum(ratio(i) * log2(ratio(i) * lambda(i))),
        
    where r_i is the firing rate of a neuron at bin i, r_mean is the mean rate of
    the neuron, time at bin i is the time spent at bin i, total time is the total
    time of recording, and log2 is the logarithm base 2.
    
    Refer to
    --------
    WE. Skaggs, et al. Theta phase precession in hippocampal neuronal populations
    and the compression of temporal sequences. Hippocampus. 1996.
    https://doi.org/10.1002/(SICI)1098-1063(1996)6:2<149::AID-HIPO6>3.0.CO;2-K
    
    Parameters
    ----------
    spikes: SpikeTrain | KilosortSpikeTrain
        The spike train of n neurons.
    rate_map: TuningCurve
        The rate map of n neurons.
    fps: float
        The frames per second of recording (50 Hz by default).
    t_interv_limits: float
        The maximum time interval allowed between two frames.
        By default, it is set to 1000 / fps * 2. (unit: ms)
    """
    # Thanks for @Ang Li
    logArg = (rate_map.T / mean_rate).T
    logArg[np.where(logArg == 0)] = 1 # keep argument in log non-zero
    
    occu_time_norm = occu_time / np.nansum(occu_time)
    
    return np.nansum(occu_time_norm * np.log2(logArg) * logArg, axis = 1)