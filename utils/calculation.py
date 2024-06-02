"""
Compute spatial information, grid score, mean vector length, etc.
"""

import numpy as np
from mazepy.datastruc import SpikeTrain, TuningCurve

def calc_infomation(
    spikes: SpikeTrain, 
    rate_map: TuningCurve,
    fps: float = 50,
    t_interv_limits: float = 100
) -> float:
    """ 
    Calculate the spatial information of neurons
    
    Parameters
    ----------
    spikes: SpikeTrain
        The spike train of n neurons.
    rate_map: TuningCurve
        The rate map of n neurons.
    fps: float
        The frame per second of recording (50 by default).
    t_interv_limits: float
        The maximum time interval allowed between two frames.
    """
    dt = np.append(np.ediff1d(spikes.time_stamp), 1000 / fps)
    dt[dt > t_interv_limits] = t_interv_limits
    
    mean_rate = np.nansum(spikes.activity, axis = 1) / t_total # mean firing rate
    logArg = (rate_map.T / mean_rate).T
    logArg[np.where(logArg == 0)] = 1 # keep argument in log non-zero