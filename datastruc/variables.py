from numpy import ndarray
import numpy as np
from typing import Optional, Union, Any

from mazepy.basic.conversion import value_to_bin
from .exceptions import DtypeError
from .exceptions import DimensionError

class VariableBin(np.ndarray):
    """
    Represents a binned variable, such as time, position, or sensory gradients.

    Parameters
    ----------
    input_array : array_like
        An array-like object which will be converted to an ndarray. 
        Elements must be integers.
    
    Methods
    -------
    get_bin_center(xmin, xmax, nbin)
        Returns the center of the bins.

    Raises
    ------
    TypeError
        If the input array is not of an integer type.
    """

    def __new__(cls, input_array: ndarray) -> 'VariableBin':
        arr = np.asarray(input_array)
        # Ensure the input is an array of integers
        if not np.issubdtype(arr.dtype, np.integer):
            raise DtypeError(
                f"Bins must be an integer array, but {arr.dtype} is given."
            )

        # Use view to create a new array instance of our type (VariableBin)
        obj = arr.view(cls)
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        pass
    
    @staticmethod
    def get_bin_center(xmin: float, xmax: float, nbin: int) -> np.ndarray:
        bin_widths = (xmax - xmin) / nbin
        left_edges = np.arange(xmin, xmax, bin_widths)
        return left_edges + bin_widths / 2

class Variable1D(ndarray):
    """
    Represents a one-dimensional variable, such as time or a single spatial 
    dimension.

    Parameters
    ----------
    input_array : array_like
        Input data to initialize the ndarray.
    meaning : str, optional
        Description or meaning of the data.

    Attributes
    ----------
    meaning : Optional[str]
        The semantic meaning attached to the variable.
    x : ndarray
        The value of the variable.

    Methods
    -------
    to_bin(xmin, xmax, nbin)
        Transform the value of the variable into bin indices.

    Examples
    --------
    >>> x = Variable1D(np.array([0, 1, 2, 3, 4, 5]), meaning="time")
    >>> print(x.meaning)
    time
    >>> print(x.x)
    [0 1 2 3 4 5]
    """
    meaning: Optional[str]

    def __new__(
        cls, 
        input_array: Any, 
        meaning: Optional[str] = None
    ) -> 'Variable1D':
        arr = np.asarray(input_array)

        if arr.ndim != 1:
            raise DimensionError(arr.ndim, "The array must be 1D.")

        obj = arr.view(cls)
        obj.meaning = meaning
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None: 
            return
        self.meaning = getattr(obj, 'meaning', None)

    @property
    def x(self) -> ndarray:
        return self

    def to_bin(
        self, 
        xmin: float,
        xmax: float,
        nbin: int
    ) -> VariableBin:
        """
        Transform the value of the variable into bin indices.

        Parameters
        ----------
        xmin : float, optional
            Minimum value for binning, defaults to 0.
        xmax : float
            Maximum value for binning.
        nbin : int
            Number of bins.

        Returns
        -------
        VariableBin
            A VariableBin instance containing the transformed data.

        See Also
        --------
        value_to_bin : Function used to calculate the bin indices.

        Examples
        --------
        >>> x = Variable1D(np.array([0.1, 1.2, 2.3, 1.1, 2.4])
        >>> nbin = 5
        >>> xmax = 5
        >>> xmin = 0
        >>> value_to_bin(x, xmin, xmax, nbin)
        array([0, 1, 2, 1, 2], dtype=int64)
        """
        return VariableBin(value_to_bin(self, nbin, xmax, xmin))
    
class Variable2D(ndarray):
    """
    Represents a 2D variable, such as time, 2-D position, or sensory gradients.

    Parameters
    ----------
    input_array : array_like
        Two-column array where each column represents one dimension of the 
        variable.
    meaning : Union[str, tuple, None], optional
        The physical or conceptual meaning of the variable.

    Raises
    ------
    ValueError
        If the input array is not two-column. If it is two-row, then it will be
        automatically converted to two-column via transposing.

    Attributes
    ----------
    meaning : Union[str, tuple, None]
        Description of what the variable represents.

    Methods
    -------
    x : ndarray
        The values of the first variable or dimension.
    y : ndarray
        The values of the second variable or dimension.
    to_bin(xmin, xmax, xnbin, ymin, ymax, ynbin)
        Transform the value of the variable into bin indices.

    Examples
    --------
    >>> x = Variable2D(
        np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
        meaning=('x', 'y')
    )
    >>> print(x.meaning)
    ('x', 'y')
    >>> print(x.x)
    [0 1 2 3]
    >>> print(x.y)
    [0 1 2 3]
    """
    meaning: Union[str, tuple[str, str], None]

    def __new__(
        cls,
        input_array: ndarray, 
        meaning: Union[str, tuple[str, str], None] = None
    ) -> 'Variable2D':
        arr = np.asarray(input_array)
        if arr.shape[1] != 2 or arr.shape[0] != 2:
            raise DimensionError(
                f"Input array must be two-column, but {arr.shape} is given."
            )
        elif arr.shape[1] != 2 or arr.shape[0] == 2:
            arr = arr.T

        if isinstance(meaning, tuple):
            if len(meaning) != 2:
                raise DimensionError(
                    "Meaning must be a tuple of length 2 or string, but "
                    f"{len(meaning)} is given."
                )

        obj = arr.view(cls)
        obj.meaning = meaning
        return obj
    
    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None:
            return
        self.meaning = getattr(obj, 'meaning', None)

    @property
    def x(self) -> Variable1D:
        if isinstance(self.meaning, tuple):
            return Variable1D(self[:, 0], meaning=self.meaning[0])
        else:
            return Variable1D(self[:, 0], meaning=self.meaning)
        
    @property
    def y(self) -> Variable1D:
        if isinstance(self.meaning, tuple):
            return Variable1D(self[:, 1], meaning=self.meaning[1])
        else:
            return Variable1D(self[:, 1], meaning=self.meaning)
        
    def to_bin(
        self, 
        xmin: float,
        xmax: float,
        xnbin: int,
        ymin: float,
        ymax: float,
        ynbin: int
    ) -> VariableBin:
        """
        Transform the 2D variable's x and y components into discrete bin 
        indices based on specified ranges and bin counts.

        This method calculates bin indices for each dimension by applying a 
        linear transformation based on the provided min and max values and 
        the total number of bins. It then combines these indices into a single 
        bin index for each data point in the dataset.

        Parameters
        ----------
        xmin : float
            Minimum value of the first dimension to be binned.
        xmax : float
            Maximum value of the first dimension to be binned.
        xnbin : int
            Number of bins for the first dimension.
        ymin : float
            Minimum value of the second dimension to be binned.
        ymax : float
            Maximum value of the second dimension to be binned.
        ynbin : int
            Number of bins for the second dimension.

        Returns
        -------
        VariableBin
            A new instance of `VariableBin` containing the combined bin indices
            for each point. The bin index is computed as:
                `xbins + xnbin * (ybins - 1)`, 
            ensuring a unique index for each (x, y) pair.

        See Also
        --------
        np.digitize : NumPy function used to "bin" the values into discrete bin 
        indices.

        Examples
        --------
        Assuming `value_to_bin` is implemented and handles the inputs correctly:

        >>> variable = Variable3D(
        ...     np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 7.5]])
        ... )
        >>> bin_indices = variable.to_bin(*(1, 4, 3), *(4, 8, 4))
        >>> print(bin_indices)
        VariableBin([])
        """
        xbins = value_to_bin(self[:, 0], xmin, xmax, xnbin)
        ybins = value_to_bin(self[:, 1], ymin, ymax, ynbin)
        return VariableBin(xbins + xnbin * (ybins - 1))
    
class Variable3D:
    """
    Represents a 3D variable, such as 3D position, velocity in 3D space, etc.

    Parameters
    ----------
    input_array : array_like
        Either a single (N, 3) array or three separate arrays (N,).
    meaning : Union[str, tuple, None], optional
        Descriptive information or physical meaning of the variable.

    Attributes
    ----------
    meaning : Union[str, tuple, None]
        Description or meaning of the variable.

    Methods
    -------
    x : ndarray
        The values of the first variable or dimension.
    y : ndarray
        The values of the second variable or dimension.
    z : ndarray
        The values of the third variable or dimension.
    to_bin(xbin, ybin, zbin, xmax, ymax, zmax, xmin, ymin, zmin)
    """
    meaning: Union[str, tuple[str, str, str], None]

    def __new__(
        cls,
        input_array: ndarray, 
        meaning: Union[str, tuple[str, str, str], None] = None
    ) -> 'Variable3D':
        arr = np.asarray(input_array)
        if arr.shape[1] != 3 or arr.shape[0] != 3:
            raise DimensionError(
                f"Input array must be three-column, but {arr.shape} is given."
            )
        elif arr.shape[1] != 3 or arr.shape[0] == 3:
            arr = arr.T

        obj = arr.view(cls)

        if isinstance(meaning, tuple):
            if len(meaning) != 3:
                raise DimensionError(
                    "Meaning must be a tuple of length 3 or string, but "
                    f"{len(meaning)} is given."
                )
            
        obj.meaning = meaning
        return obj
    
    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None:
            return
        self.meaning = getattr(obj, 'meaning', None)

    @property
    def x(self) -> Variable1D:
        if isinstance(self.meaning, tuple):
            return Variable1D(self[:, 0], meaning=self.meaning[0])
        else:
            return Variable1D(self[:, 0], meaning=self.meaning)
    
    @property
    def y(self) -> Variable1D:
        if isinstance(self.meaning, tuple):
            return Variable1D(self[:, 1], meaning=self.meaning[1])
        else:
            return Variable1D(self[:, 1], meaning=self.meaning)
    
    @property
    def z(self) -> Variable1D:
        if isinstance(self.meaning, tuple):
            return Variable1D(self[:, 2], meaning=self.meaning[2])
        else:
            return Variable1D(self[:, 2], meaning=self.meaning)

    def to_bin(
        self,
        xmin: float,
        xmax: float,
        xnbin: int,
        ymin: float,
        ymax: float,
        ynbin: int,
        zmin: float,
        zmax: float,
        znbin: int
    ) -> VariableBin:
        """
        Transform the 3D variable's x, y, and z components into discrete bin 
        indices based on specified ranges and bin counts.

        This method calculates bin indices for each dimension by applying a 
        linear transformation based on the provided min and max values and 
        the total number of bins. It then combines these indices into a single 
        bin index for each data point in the dataset.

        Parameters
        ----------
        xmin : float
            Minimum value of the first dimension to be binned.
        xmax : float
            Maximum value of the first dimension to be binned.
        xnbin : int
            Number of bins for the first dimension.
        ymin : float
            Minimum value of the second dimension to be binned.
        ymax : float
            Maximum value of the second dimension to be binned.
        ynbin : int
            Number of bins for the second dimension.
        zmin : float
            Minimum value of the third dimension to be binned.
        zmax : float
            Maximum value of the third dimension to be binned.
        znbin : int
            Number of bins for the third dimension.

        Returns
        -------
        VariableBin
            A new instance of `VariableBin` containing the combined bin indices
            for each point. The bin index is computed as:
                `xbins + xnbin * (ybins - 1) + xnbin * ynbin * (zbins - 1)`, 
            ensuring a unique index for each (x, y, z) pair.

        See Also
        --------
        np.digitize : NumPy function used to "bin" the values into discrete bin 
        indices.

        Examples
        --------
        >>> from mazepy.datastruc.variables import Variable3D
        >>> variable = Variable3D(
        ...     np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 7.5], [8.5, 9.5, 10.5]])
        ... )
        >>> bin_indices = variable.to_bin(*(1, 4, 3), *(4, 8, 4), *(8, 11, 3))
        >>> print(bin_indices)
        VariableBin([])
        """
        xbins = value_to_bin(self.x, xmin, xmax, xnbin)
        ybins = value_to_bin(self.y, ymin, ymax, ynbin)
        zbins = value_to_bin(self.z, zmin, zmax, znbin)
        return VariableBin(
            xbins + xnbin * (ybins - 1) + xnbin * ynbin * (zbins - 1)
        )

Variables = Union[Variable1D, Variable2D, Variable3D, VariableBin, ndarray]


if __name__ == "__main__":
    import doctest
    doctest.testmod()