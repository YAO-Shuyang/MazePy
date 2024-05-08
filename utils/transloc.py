import numpy as np

def value_to_bin(
    x: np.ndarray, 
    nbin: int, 
    xmax: float, 
    xmin: float = 0
) -> np.ndarray:
    """
    Transform value to bin index based on specified range and number of bins.

    Parameters
    ----------
    x : np.ndarray
        The values to be transformed into bin indices.
    nbin : int
        The total number of bins.
    xmax : float
        The maximum value in the range of x.
    xmin : float
        The minimum value in the range of x, default is 0.

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
    >>> value_to_bin(x, nbin, xmax, xmin)
    array([0, 1, 2, 1, 2, 3, 4, 4, 3], dtype=int64)
    """
    # Implementation assumes linear spacing between xmin and xmax
    bins = np.linspace(xmin, xmax*1.0001, num=nbin + 1)  
    # np.digitize returns indices starting from 1
    bin_indices = np.digitize(x, bins) - 1  
    # Ensure bin indices are within valid range
    bin_indices = np.clip(bin_indices, 0, nbin - 1).astype(np.int64) + 1
    return bin_indices

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    