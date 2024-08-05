from re import X
from networkx import center
from numpy import isin, ndarray
import numpy as np
from typing import Optional, Union, Any

from mazepy.basic.convert import value_to_bin

from mazepy.datastruc.exceptions import DtypeError, DimensionError

class _VariableBase(ndarray):
    def __new__(cls, input_array: ndarray) -> '_VariableBase':
        obj = np.asarray(input_array).view(cls)
        return obj
    
    def __array_finalize__(self, obj: Optional[ndarray]) -> None:
        pass
    
    def __init__(self, input_array: ndarray) -> None:
        pass
    
    def to_array(self) -> ndarray:
        return np.asarray(self)

class VariableBin(_VariableBase):
    """
    Represents a binned variable, such as time, position, or sensory gradients.

    Parameters
    ----------
    input_array : array_like
        An array-like object which will be converted to an ndarray. 
        Elements must be integers.
    
    Methods
    -------
    @staticmethod
    get_bin_center1d(xmin, xmax, nbins)
        Returns the center of the bins.

    Raises
    ------
    TypeError
        If the input array is not of an integer type.
    """

    def __new__(cls, input_array: ndarray) -> 'VariableBin':
        if isinstance(input_array, ndarray) == False:
            arr = np.asarray(input_array)
        else:
            arr = input_array
            
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
    
    def __init__(self, input_array: ndarray) -> None:
        super().__init__(input_array)
    
    @staticmethod
    def get_bin_center1d(xmin: float, xmax: float, xnbins: int) -> np.ndarray:
        """
        Get the center of the bins based on the min and max values and the
        number of bins. The bin width is assumed to be uniform and uniquely
        determined by the number of bins and the range by:

            bin_width = (xmax - xmin) / nbins

        Parameters
        ----------
        xmin : float
            The minimum value of the range.
        xmax : float
            The maximum value of the range.
        xnbins : int
            The number of bins.

        Returns
        -------
        np.ndarray
            The center of the bins.
            
        Examples
        --------
        >>> from mazepy.datastruc.variables import VariableBin
        >>> VariableBin.get_bin_center1d(0, 10, 5)
        array([1., 3., 5., 7., 9.])
        >>> # Another way to input, which is recommended when get the bin
        >>> # centers for 2D or 3D variables.
        >>> VariableBin.get_bin_center1d(*(0, 10, 5))
        array([1., 3., 5., 7., 9.])
        """
        if xmax <= xmin:
            raise ValueError(
                f"xmax must be greater than xmin, but {xmax} <= {xmin}."
            )
            
        bin_widths = (xmax - xmin) / xnbins
        left_edges = np.arange(xmin, xmax, bin_widths)
        return left_edges + bin_widths / 2
    
    @staticmethod
    def get_bin_center2d(
        xmin: float, 
        xmax: float, 
        xnbins: int, 
        ymin: float, 
        ymax: float, 
        ynbins: int
    ) -> np.ndarray:
        """
        Get the center of the bins based on the min and max values and the
        number of bins. The bin width is assumed to be uniform and uniquely
        determined by the number of bins and the range by:

            bin_width_x = (xmax - xmin) / xnbins
            bin_width_y = (ymax - ymin) / ynbins

        Parameters
        ----------
        xmin : float
            The minimum value of the range.
        xmax : float
            The maximum value of the range.
        xnbins : int
            The number of bins in the x direction.
        ymin : float
            The minimum value of the range.
        ymax : float
            The maximum value of the range.
        ynbins : int
            The number of bins in the y direction.

        Returns
        -------
        np.ndarray, with shape (xnbins * ynbins, 2)
            The center of the bins.
            
        Examples
        --------
        >>> from mazepy.datastruc.variables import VariableBin
        >>> VariableBin.get_bin_center2d(0, 10, 2, 0, 2, 2)
        array([[2.5, 0.5],
               [7.5, 0.5],
               [2.5, 1.5],
               [7.5, 1.5]])
        >>> # Recommmended way to input, grouping the parameters for each
        >>> # dimension for clarity.
        >>> VariableBin.get_bin_center2d(*(0, 10, 2), *(0, 2, 2))
        array([[2.5, 0.5],
               [7.5, 0.5],
               [2.5, 1.5],
               [7.5, 1.5]])
        """
        centers_x = VariableBin.get_bin_center1d(xmin, xmax, xnbins)
        centers_y = VariableBin.get_bin_center1d(ymin, ymax, ynbins)
        centers_x, centers_y = np.meshgrid(centers_x, centers_y)
        return np.vstack((np.ravel(centers_x), np.ravel(centers_y))).T
    
    @staticmethod
    def get_bin_center3d(
        xmin: float, 
        xmax: float, 
        xnbins: int, 
        ymin: float, 
        ymax: float, 
        ynbins: int, 
        zmin: float, 
        zmax: float, 
        znbins: int
    ) -> np.ndarray:
        """
        Get the center of the bins based on the min and max values and the
        number of bins. The bin width is assumed to be uniform and uniquely
        determined by the number of bins and the range by:

            bin_width_x = (xmax - xmin) / xnbins
            bin_width_y = (ymax - ymin) / ynbins
            bin_width_z = (zmax - zmin) / znbins

        Parameters
        ----------
        xmin : float
            The minimum value of the range.
        xmax : float
            The maximum value of the range.
        xnbins : int
            The number of bins in the x direction.
        ymin : float
            The minimum value of the range.
        ymax : float
            The maximum value of the range.
        ynbins : int
            The number of bins in the y direction.
        zmin : float
            The minimum value of the range.
        zmax : float
            The maximum value of the range.
        znbins : int
            The number of bins in the z direction.

        Returns
        -------
        np.ndarray, with shape (xnbins * ynbins * znbins, 3)
            The center of the bins.
            
        Examples
        --------
        >>> from mazepy.datastruc.variables import VariableBin
        >>> VariableBin.get_bin_center3d(0, 10, 2, 0, 10, 2, 0, 4, 2)
        array([[2.5, 2.5, 1. ],
               [7.5, 2.5, 1. ],
               [2.5, 7.5, 1. ],
               [7.5, 7.5, 1. ],
               [2.5, 2.5, 3. ],
               [7.5, 2.5, 3. ],
               [2.5, 7.5, 3. ],
               [7.5, 7.5, 3. ]])
        >>> # Recommmended way to input, grouping the parameters for each
        >>> # dimension for clarity.
        >>> VariableBin.get_bin_center3d(*(0, 10, 2), *(0, 2, 2), *(0, 4, 2))
        array([[2.5, 0.5, 1. ],
               [7.5, 0.5, 1. ],
               [2.5, 1.5, 1. ],
               [7.5, 1.5, 1. ],
               [2.5, 0.5, 3. ],
               [7.5, 0.5, 3. ],
               [2.5, 1.5, 3. ],
               [7.5, 1.5, 3. ]])
        """
        left_centers_x = VariableBin.get_bin_center1d(xmin, xmax, xnbins)
        left_centers_y = VariableBin.get_bin_center1d(ymin, ymax, ynbins)
        left_centers_z = VariableBin.get_bin_center1d(zmin, zmax, znbins)
        # This order is necessary, I just have no idea why.
        centers_y, centers_z, centers_x = np.meshgrid(
            left_centers_y,
            left_centers_z,
            left_centers_x
        )
        return np.vstack((
            np.ravel(centers_x), 
            np.ravel(centers_y), 
            np.ravel(centers_z)
        )).T
        

class Variable1D(_VariableBase):
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
    to_bin(xmin, xmax, nbins)
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
        if isinstance(input_array, ndarray) == False:
            arr = np.asarray(input_array)
        else:
            arr = input_array

        if arr.ndim != 1:
            raise DimensionError(arr.ndim, "The array must be 1D.")

        return arr.view(cls)

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        if obj is None: 
            return
        self.meaning = getattr(obj, 'meaning', None)
        
    def __init__(self, input_array: ndarray, meaning: Optional[str] = None) -> None:
        super().__init__(input_array)
        self.meaning = meaning
        
    @property
    def x(self) -> ndarray:
        return self

    def to_bin(
        self, 
        xmin: float,
        xmax: float,
        nbins: int
    ) -> VariableBin:
        """
        Binned Index starts from `0` to `(nbins - 1)`.

        Transform the value of the variable into bin indices.

        Parameters
        ----------
        xmin : float, optional
            Minimum value for binning, defaults to 0.
        xmax : float
            Maximum value for binning.
        nbins : int
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
        >>> x = Variable1D([0.1, 1.2, 2.3, 1.1, 2.4])
        >>> nbins = 5
        >>> xmax = 5
        >>> xmin = 0
        >>> value_to_bin(x, xmin, xmax, nbins)
        array([0, 1, 2, 1, 2], dtype=int64)
        """
        return VariableBin(value_to_bin(self, xmin, xmax, nbins))
    
class Variable2D(_VariableBase):
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
    to_bin(xmin, xmax, xnbins, ymin, ymax, ynbins)
        Transform the value of the variable into bin indices.

    Examples
    --------
    >>> x = Variable2D([[0, 1, 2, 3], [1, 2, 3, 4]], meaning=('x', 'y'))
    >>> x.meaning
    ('x', 'y')
    >>> x.x
    Variable1D([0, 1, 2, 3])
    >>> x.y
    Variable1D([1, 2, 3, 4])
    """
    meaning: Union[str, tuple[str, str], None]

    def __new__(
        cls,
        input_array: ndarray, 
        meaning: Union[str, tuple[str, str], None] = None
    ) -> 'Variable2D':
        if isinstance(input_array, ndarray) == False:
            arr = np.asarray(input_array)
        else:
            arr = input_array
            
        if arr.shape[1] != 2 and arr.shape[0] != 2:
            raise DimensionError(
                f"Input array must be two-column, but {arr.shape} is given."
            )
        elif arr.shape[1] != 2 and arr.shape[0] == 2:
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
        
    def __init__(
        self,
        input_array: ndarray, 
        meaning: Union[str, tuple[str, str], None] = None
    ) -> None:
        super().__init__(input_array)

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
        xnbins: int,
        ymin: float,
        ymax: float,
        ynbins: int
    ) -> VariableBin:
        """
        Binned Index starts from `0` to `(xnbins * ynbins - 1)`.
            
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
        xnbins : int
            Number of bins for the first dimension.
        ymin : float
            Minimum value of the second dimension to be binned.
        ymax : float
            Maximum value of the second dimension to be binned.
        ynbins : int
            Number of bins for the second dimension.

        Returns
        -------
        VariableBin
            A new instance of `VariableBin` containing the combined bin indices
            for each point. The bin index is computed as:
                `xbins + xnbins * (ybins - 1)`, 
            ensuring a unique index for each (x, y) pair.

        See Also
        --------
        np.digitize : NumPy function used to "bin" the values into discrete bin 
        indices.

        Examples
        --------
        Assuming `value_to_bin` is implemented and handles the inputs correctly:

        >>> variable = Variable3D([
        ...     [1.5, 2.5, 3.5], [4.5, 5.5, 7.5], [6.5, 7.5, 8.5]
        ... ])
        >>> bin_indices = variable.to_bin(*(1, 4, 3), *(4, 8, 4), *(6, 9, 3))
        >>> bin_indices
        VariableBin([ 0, 17, 35], dtype=int64)
        """
        xbins = value_to_bin(self[:, 0], xmin, xmax, xnbins)
        ybins = value_to_bin(self[:, 1], ymin, ymax, ynbins)
        return VariableBin(xbins + xnbins * (ybins - 1))
    
class Variable3D(_VariableBase):
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
        if isinstance(input_array, np.ndarray) == False:
            arr = np.asarray(input_array)
        else:
            arr = input_array
            
        if arr.shape[1] != 3 and arr.shape[0] != 3:
            raise DimensionError(
                f"Input array must be three-column, but {arr.shape} is given."
            )
        elif arr.shape[1] != 3 and arr.shape[0] == 3:
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
        
    def __init__(
        self,
        input_array: ndarray, 
        meaning: Union[str, tuple[str, str, str], None] = None
    ) -> None:
        super().__init__(input_array)

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
        xnbins: int,
        ymin: float,
        ymax: float,
        ynbins: int,
        zmin: float,
        zmax: float,
        znbins: int
    ) -> VariableBin:
        """
        Binned Index starts from `0` to `(xnbins * ynbins * znbins - 1)`.
        
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
        xnbins : int
            Number of bins for the first dimension.
        ymin : float
            Minimum value of the second dimension to be binned.
        ymax : float
            Maximum value of the second dimension to be binned.
        ynbins : int
            Number of bins for the second dimension.
        zmin : float
            Minimum value of the third dimension to be binned.
        zmax : float
            Maximum value of the third dimension to be binned.
        znbins : int
            Number of bins for the third dimension.

        Returns
        -------
        VariableBin
            A new instance of `VariableBin` containing the combined bin indices
            for each point. The bin index is computed as:
                `xbins + xnbins * (ybins - 1) + xnbins * ynbins * (zbins - 1)`, 
            ensuring a unique index for each (x, y, z) pair.

        See Also
        --------
        np.digitize : NumPy function used to "bin" the values into discrete bin 
        indices.

        Examples
        --------
        >>> from mazepy.datastruc.variables import Variable3D
        >>> variable = Variable3D([
        ...     [1.5, 2.5, 3.5], 
        ...     [4.5, 5.5, 7.5], 
        ...     [8.5, 9.5, 10.5]
        ... ])
        >>> bin_indices = variable.to_bin(*(1, 4, 3), *(4, 8, 4), *(8, 11, 3))
        >>> bin_indices
        VariableBin([ 0,  5, 35], dtype=int64)
        """
        xbins = value_to_bin(self.x, xmin, xmax, xnbins)
        ybins = value_to_bin(self.y, ymin, ymax, ynbins)
        zbins = value_to_bin(self.z, zmin, zmax, znbins)
        return VariableBin(
            xbins + xnbins * ybins + xnbins * ynbins * zbins
        )

Variables = Union[Variable1D, Variable2D, Variable3D, VariableBin, ndarray]


if __name__ == "__main__":
    import doctest
    doctest.testmod()