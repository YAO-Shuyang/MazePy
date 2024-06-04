from typing import Optional
import numpy as np
from mazepy.basic._time_sync import _coordinate_recording_time

def value_to_bin(
    x: np.ndarray, 
    xmin: float, 
    xmax: float, 
    nbin: int
) -> np.ndarray:
    """
    Transform value to bin index based on specified range and number of bins.

    Parameters
    ----------
    x : np.ndarray
        The values to be transformed into bin indices.
    xmin : float
        The minimum value in the range of x
    xmax : float
        The maximum value in the range of x.
    nbin : int
        The total number of bins.

    Returns
    -------
    np.ndarray
        An array of bin indices corresponding to the values in x
        dtype=np.int64.
        
    Examples
    --------
    >>> x = np.array([0.1, 1.2, 2.3, 1.1, 2.4, 3.5, 4.2, 4.9, 3.6])
    >>> nbin = 5
    >>> xmax = 5
    >>> xmin = 0
    >>> value_to_bin(x, xmin, xmax, nbin)
    array([0, 1, 2, 1, 2, 3, 4, 4, 3], dtype=int64)
    """
    # Implementation assumes linear spacing between xmin and xmax
    bins = np.linspace(xmin, xmax*1.0001, num=nbin + 1)  
    # np.digitize returns indices starting from 1
    bin_indices = np.digitize(x, bins) - 1  
    # Ensure bin indices are within valid range
    bin_indices = np.clip(bin_indices, 0, nbin - 1).astype(np.int64)
    return bin_indices

def coordinate_recording_time(
    source_time: np.ndarray, 
    target_time: np.ndarray
) -> np.ndarray:
    """
    Coordinate recording time of behavioral data and neural data.

    Parameters
    ----------
    source_time : np.ndarray
        The time to be converted (typically the neural data).
    target_time : np.ndarray
        The targeted time (typically the behavioral data). 

    Returns
    -------
    np.ndarray
        The index of each source_time in target_time. The same length as
        source_time.

    Examples
    --------
    >>> source_time = np.array([0, 33, 67, 100, 133, 167, 200, 233, 267, 300])
    >>> target_time = np.array([0, 50, 100, 150, 200, 250, 300])
    >>> coordinate_recording_time(source_time, target_time)
    array([0, 1, 1, 2, 3, 3, 4, 5, 5, 6], dtype=int64)
    """
    return _coordinate_recording_time(
        source_time.astype(np.float64), 
        target_time.astype(np.float64)
    )
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    