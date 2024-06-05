"""
Provides functions for calculation.
"""

import numpy as np
from typing import Optional, Union
from mazepy.datastruc import SpikeTrain, TuningCurve, KilosortSpikeTrain

def calc_information(
    spikes: Union[SpikeTrain, KilosortSpikeTrain], 
    rate_map: Union[TuningCurve, np.ndarray], 
    t_total
) -> float:
    """ 
    Calculate the spatial information of neurons
    
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
    mean_rate = np.nansum(spikes, axis = 1) / t_total # mean firing rate
    logArg = (rate_map.T / mean_rate).T
    logArg[np.where(logArg == 0)] = 1 # keep argument in log non-zero