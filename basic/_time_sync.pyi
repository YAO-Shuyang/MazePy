import numpy as np

def _coordinate_recording_time(
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
    ...