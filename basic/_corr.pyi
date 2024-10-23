import numpy as np

def pearsonr(
    x: np.ndarray,
    y: np.ndarray
) -> tuple:
    """
    Computes Pearson Correlation
    
    Parameter
    ---------
    x: np.ndarray,
        The first vector
    y: np.ndarray
        The second vector
    
    Returns
    -------
    correlation, cov_xy
    """
    ...
    
def pearsonr_pairwise(
    data: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    Compute column-wise or row-wise pearson correlation
    
    Parameters
    ----------
    data: np.ndarray, 2-d
        The vectors to compute correlation
    axis: int, by default = 2
        The axis to compute pearson correlation
    """
    ...
