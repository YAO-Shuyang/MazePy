from signal import NSIG
from numpy import ndarray
import numpy as np
from typing import Optional,Callable

"""
Define data structure for kernels, for the purpose of convolution and 
smoothing.
"""

def get_uniform_kernel(n: int) -> ndarray:
    """
    Returns a uniform kernel with `n` elements.

    Parameters
    ----------
    n: int
        The width of the kernel.

    Returns
    -------
    ndarray
        A uniform kernel where each element is `1/n`.

    Examples
    -------- 
    >>> get_uniform_kernel(5)
    array([0.2, 0.2, 0.2, 0.2, 0.2])
    """
    return np.ones(n) / n

def get_gaussian_kernel(n: int, sigma: float) -> ndarray:
    """
    Returns a gaussian kernel with n elements.

    Parameters
    ----------
    n: int
        The width of the kernel.
    sigma: float
        The standard deviation of the gaussian kernel.

    Returns
    -------
    ndarray
        A gaussian kernel with n (when n is odd) or n-1 (when n is even) 
        elements to make the kernel symmetric.

    Examples
    --------
    >>> from mazepy.datastruc.kernel import get_gaussian_kernel
    >>> import numpy as np
    >>> 
    >>> get_gaussian_kernel(5, 1)
    array([0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868])
    """
    # symmetric
    half_n = int((n-1) / 2)
    kernel = np.exp(-np.arange(-half_n, half_n+1)**2 / (2*sigma**2))
    return kernel / np.sum(kernel)

class _Kernel1d(ndarray):
    """
    A base class for creating 1D kernel arrays, subclass of ndarray.

    This class should not be instantiated directly.
    """
    def __new__(cls, n: int, func: Callable, *func_args) -> '_Kernel1d': 
        arr = np.asarray(func(n, *func_args))
        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj: Optional[ndarray]) -> None:
        pass

class UniformKernel1d(_Kernel1d):
    """
    Represents a uniform 1D kernel where each element has equal weight.

    Attributes
    ----------
    n : int
        The width (number of elements) of the kernel.

    Methods
    -------
    __new__
        Creates an instance of UniformKernel1d using a uniform distribution.

    Examples
    --------
    >>> kernel = UniformKernel1d(5)
    >>> print(kernel)
    [0.2 0.2 0.2 0.2 0.2]
    """
    def __new__(cls, n: int) -> 'UniformKernel1d':
        return super().__new__(cls, n, get_uniform_kernel)

class GaussianKernel1d(_Kernel1d):
    """
    Represents a 1D Gaussian kernel used for operations like smoothing.

    Attributes
    ----------
    n : int
        The width of the kernel.
    sigma : float
        The standard deviation of the Gaussian distribution.

    Methods
    -------
    __new__
        Creates an instance of GaussianKernel1d using a Gaussian distribution.

    Examples
    --------
    >>> kernel = GaussianKernel1d(5, 1)
    >>> print(kernel)
    [0.05448868 0.24420134 0.40261995 0.24420134 0.05448868]
    """
    def __new__(cls, n: int, sigma: float) -> 'GaussianKernel1d':
        return super().__new__(cls, n, get_gaussian_kernel, sigma)


class _Kernel2d(ndarray):
    """
    A base class for creating 2D kernel arrays, subclass of ndarray.

    This class should not be instantiated directly.
    """
    def __new__(cls, input_array: ndarray) -> '_Kernel2d': 
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj: Optional[ndarray]) -> None:  
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()