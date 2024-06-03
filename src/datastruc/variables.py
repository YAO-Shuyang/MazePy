import array
from ast import Return
from matplotlib import axis
from numpy import isin, ndarray
import numpy as np
from typing import Optional, Union, Callable, Any

from mazepy.basic.conversion import value_to_bin
from mazepy.datastruc.kernel import GaussianKernel1d, UniformKernel1d
from mazepy.datastruc.exceptions import DtypeError, ViolatedConstraintError
from mazepy.datastruc.exceptions import DimensionMismatchError, DimensionError
import warnings
from scipy.stats import binned_statistic
from urllib3 import Retry

Kernels = Union[GaussianKernel1d, UniformKernel1d, ndarray]

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

class TuningCurve(ndarray):
    """
    A class for representing tuning curves, typically used to describe the 
    firing behavior of neurons across different conditions or stimuli.

    The tuning curve data is stored as a NumPy array where each row represents 
    a neuron and each column represents a different stimulus or condition. The 
    elements of the array represent the firing rates of the neurons.

    Parameters
    ----------
    firing_rate : ndarray, shape (n_neuron, n_condition)
        An array-like object containing the firing rates of neurons. This can 
        be a 2D array where each row corresponds to a neuron and each column 
        corresponds to a different condition or stimulus, for instance, a 
        different spatial bin.
    occu_time : ndarray, shape (n_condition, )
        An array-like object containing the time spent in each condition or 
        stimulus. This should be an 1D array with the same number of elements as 
        the number of columns in the `firing_rate` array.

    Methods
    -------
    n_neuron
        Returns the number of neurons represented in the tuning curve.
    get_argpeaks
        Returns the indices of the peak firing rates for each neuron.
    get_peaks
        Returns the peak firing rates for each neuron.
    remove_nan
        Replaces NaN values in the firing rate data with zeros.

    Examples
    --------
    >>> firing_rates = np.array([[0, 2, 3], [1, 5, 2], [3, 0, 0]])
    >>> tuning_curve = TuningCurve(firing_rates)
    >>> print(tuning_curve.n_neuron)
    3
    >>> print(tuning_curve.get_argpeaks())
    [2, 1, 0]
    >>> print(tuning_curve.get_peaks())
    [3, 5, 3]
    """
    occu_time: np.ndarray

    def __new__(
        cls, 
        firing_rate: ndarray, 
        occu_time: np.ndarray
    ) -> 'TuningCurve':
        obj = np.asarray(firing_rate).view(cls)
        obj.occu_time = occu_time
        return obj
    
    def __array_finalize__(self, obj: Optional[ndarray]) -> None:
        if obj is None:
            return
        self.occu_time = getattr(obj, 'occu_time', None)
    
    @property
    def n_neuron(self) -> int:
        return self.shape[0]
    
    def get_argpeaks(self) -> ndarray:
        """
        Get the index of the peak firing rate of each neuron.
        """
        return np.nanargmax(self, axis=1)
    
    def get_peaks(self) -> ndarray:
        """
        Get the peak firing rate of each neuron.
        """
        return np.nanmax(self, axis=1)
    
    def remove_nan(self) -> None:
        """
        Remove NaN values from the array by setting them to 0.
        """
        self[np.isnan(self)] = 0

    def get_fields(self) -> list[dict]:
        # Provides Breadth First Search (BFS) implementation for candidate
        # response fields
        raise NotImplementedError
    
    def _smooth1d(self, kernel: Kernels) -> 'TuningCurve':
        """
        Smooth the tuning curve with one dimensions using a given kernel.
        """
        smoothed_curve = np.zeros_like(self)
        for i in range(self.shape[0]):
            smoothed_curve[i] = np.convolve(self[i, :], kernel)
        return TuningCurve(smoothed_curve)
    
    def _smooth2d(self, kernel: Kernels) -> 'TuningCurve':
        """
        Smooth the tuning curve with two dimensions using a given kernel.
        """
        return TuningCurve(self @ kernel)
    
    def smooth(self, kernel: Kernels) -> 'TuningCurve':
        """
        Smooth the tuning curve using a given smoothing matrix.

        Parameters
        ----------
        kernel : Kernels
            The kernel to be used for smoothing.

        Returns
        -------
        TuningCurve
            The smoothed tuning curve.
        """
        if kernel.ndim == 1:
            return self._smooth1d(kernel)
        elif kernel.ndim == 2:
            return self._smooth2d(kernel)
        else:
            raise ValueError(
                f"Kernel must be 1D or 2D, but got {kernel.ndim}D."
            )

class NeuralTrajectory(ndarray):
    """
    A class for representing temporal population vectors, typically used to 
    describe the temporal dynamics of a population of neurons, namely the 
    n-dimensional neural trajectory.

    The neural trajectory data is stored as a NumPy array where each row 
    represents a neuron and each column represents a different time bin (a
    short period of time). The shape of the array should be (n_neuron, n_time).

    Parameters
    ----------
    neural_trajectory : ndarray, shape (n_neuron, n_time)
        An array-like object containing the neural trajectory data. This should
        be a 2D array where each row corresponds to a neuron and each column 
        corresponds to a different time bin. 
    time : Union[Variable1D, ndarray], shape (n_time, )
        Unit: ms
        An array-like object containing the time data associated with the
        neural trajectory. This should be a 1D array where each element 
        corresponds to a different time bin.
    variable : Optional[VariableBin], default None, shape (n_time,) if provided
        Optional variable data associated with the neural trajectory, such as
        stimulus conditions or experimental variables.

    Raises
    ------
    DimensionMismatchError
        If the shape of the neural trajectory does not match the shape of the
        time data or the shape of the variable data, if provided.

    Methods
    -------
    n_neuron
        Returns the number of neurons constituting the neural trajectory.

    Examples
    --------
    >>> neural_trajectory = np.array([[0, 2, 3], [1, 5, 2], [3, 0, 0]])
    >>> time = np.array([0, 50, 100])
    >>> trajectory = NeuralTrajectory(neural_trajectory, time)
    >>> print(trajectory.n_neuron)
    3
    """
    time: Union[Variable1D, ndarray]
    variable: Optional[VariableBin]

    def __new__(
        cls, 
        neural_trajectory: ndarray, 
        time: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None
    ) -> 'NeuralTrajectory':
        obj = np.asarray(neural_trajectory).view(cls)
        
        if time.shape[0] != obj.shape[1]:
            raise DimensionMismatchError(
                f"Neural trajectory and time should have the same number of "
                f"time bins."
            )
        
        if variable is not None:
            if variable.shape[0] != time.shape[0]:
                raise DimensionMismatchError(
                    f"Time and variable should have the same "
                    f"number of time bins."
                )
        
        obj.time = time
        obj.variable = variable
        return obj

    def __array_finalize__(self, obj: Optional[ndarray]) -> None:
        if obj is None:
            return
        self.time = getattr(obj, 'time', None)
        self.variable = getattr(obj, 'variable', None)

class _NeuralActivity(ndarray):
    """
    A base class for representing neural activity in terms of neurons and 
    time points.
    
    This class is designed to store neural activity data along with associated 
    time and variable data. It supports operations to manage missing data and 
    ensure consistency in the dimensions of related data arrays.

    Attributes
    ----------
    time : Union[Variable1D, ndarray]
        Time data associated with the neural activity.
    variable : Optional[VariableBin]
        Optional variable data associated with the neural activity, such as
        stimulus conditions or experimental variables.

    Methods
    -------
    remove_nan():
        Removes NaN values from neural activity and corresponding time and 
        variable arrays.

    Raises
    ------
    DimensionMismatchError
        If the dimensions of the provided arrays do not match as required.
        The shape of activity should be (n_neuron, n_time), whereas the shape
        of time and, if provided, variable should be (n_time,).
    TypeError
        If the variable provided is not a VariableBin or None. You should 
        convert the variable to a VariableBin if it is not already one.
    """
    time: Union[Variable1D, ndarray]
    variable: Optional[VariableBin]

    def __new__(
        cls, 
        activity: ndarray, 
        time: ndarray, 
        variable: Optional[VariableBin] = None
    ) -> '_NeuralActivity':
        """
        Parameters
        ----------
        activity : ndarray, shape (n_neurons, n_time)
            The neural activity data.
        time : ndarray, shape (n_time,)
            The time data associated with the neural activity.
        variable : Optional[VariableBin], default None, shape (n_time,) if provided
            Optional variable data associated with the neural activity, such as
            stimulus conditions or experimental variables.
        """
        arr = np.asarray(activity)

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim > 2:
            raise DimensionMismatchError(
                len1=arr.ndim, 
                len2=2,
                message="The shape of activity should be (n_neurons, n_time),"
            )
        
        obj = arr.view(cls)
        obj.time = time
        obj.variable = variable

        if activity.shape[1] != time.shape[0]:
            raise DimensionMismatchError(
                len1=activity.shape[1], 
                len2=time.shape[0],
                message="Neural activity and time should have the same number of "
                "frames!"
            )
        
        if isinstance(variable, VariableBin) and time.shape[0] != variable.shape[0]:
            raise DimensionMismatchError(
                len1=time.shape[0], 
                len2=variable.shape[0],
                message="Time and variable should have the same number of frames!"
            )
            
        obj.remove_nan()
        return obj
    
    def __array_finalize__(self, obj: Optional[ndarray]) -> None:
        if obj is None:
            return
        
        self.time = getattr(obj, 'time', None)
        self.variable = getattr(obj, 'variable', None)
    
    @property
    def n_neuron(self) -> int:
        """
        Return the number of neurons represented in the activity data.
        """
        return self.shape[0]
    
    def remove_nan(self) -> None:
        """
        Remove NaN values from neural activity, time, and variable arrays 
        in-place.
        """
        if self.variable is not None:
            nan_indices = np.where(
                np.isnan(np.sum(self, axis=0)) |
                np.isnan(self.time) |
                np.isnan(self.variable)
            )[0]
            self.variable = np.delete(self.variable, nan_indices)
        else:
            nan_indices = np.where( 
                np.isnan(np.sum(self, axis=0)) |
                np.isnan(self.time)
            )[0]
            
        self[:] = np.delete(self, nan_indices, axis=1)
        self.time = np.delete(self.time, nan_indices)

class SpikeTrain(_NeuralActivity):
    """
    A class specifically for representing spike train data. Spike trains record
    the firing of neurons over time, typically binary data indicating the 
    presence or absence of a spike.

    Attributes
    ----------
    activity : ndarray, shape (n_neurons, n_time)
        The spike trains of each neuron, with shape (n_neurons, n_time).
        All entries should be either 1 or 0.
    time : Variable1D or ndarray, shape (n_time, )
        The time stamps for each point in the activity array, with units in 
        milliseconds.
    variable : Optional[VariableBin], shape (n_time, ) if provided
        Optionally, additional metadata corresponding to each time point, such 
        as experimental conditions or stimulus information.

    Raises
    ------
    ViolatedConstraintError
        If the activity data are not binary.
    """
    def __init__(
        self,
        activity: ndarray,
        time: Union[Variable1D, ndarray],  
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Initialize the SpikeTrain with activity data, time stamps, and 
        optional bin indices.
        """
        if not np.all(np.isin(activity, [0, 1])):
            raise ViolatedConstraintError(
                "All entries of the spike train should be either 0 or 1."
            )

        super().__init__(activity, time, variable)
    
    def _get_temp_variable(self) -> Variable1D:
        """
        
        """

    def calc_temporal_tuning_curve(
        self, 
        t_window: float,
        step_size: Optional[float] = None
    ) -> TuningCurve:
        """
        Calculate the temporal firing rate of neurons using specified windowing
        method.
        
        Parameters
        ----------
        t_window: float
            The time window in miliseconds, e.g., 400
        step_size: 
            The step size in milliseconds for moving the window; defaults to 
            twindow if None, which makes it non-overlapping.
            
        Returns
        -------
        TuningCurve
            The temporal population vector of each neuron.

        Examples
        --------
        >>> from mazepy.datastruc.variables import SpikeTrain
        >>> import numpy as np
        >>>
        >>> spikes = SpikeTrain(
        ...     # Spike Train
        ...     activity = np.array([
        ...         [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        ...         [1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        ...         [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1]
        ...     ]),
        ...     # Time stamps (unit: ms)
        ...     time = np.array([
        ...         0, 30, 32, 50, 90, 102, 110, 130, 154, 180, 200, 205, 240,
        ...         260, 300
        ...     ])
        ... )
        >>> tuning_curve = spikes.calc_temporal_tuning_curve(50)
        >>> # time bins: [0, 50], [50, 100], [100, 150], ...
        >>> tuning_curve.firing_rate
        array([[40., 20.,  0., 20., 40., 20.,  0.],
               [40.,  0., 40., 40., 20.,  0., 20.],
               [40., 40., 20., 20., 40.,  0., 20.]])
        >>> tuning_curve = spikes.calc_temporal_tuning_curve(50, 20)
        >>> # time bins: [0, 50], [20, 70], [40, 90], [60, 110], ...
        >>> tuning_curve.firing_rate
        array([[40., 40., 20.,  0.,  0.,  0., 20., 20., 20., 20., 40., 40., 40.,
                20.],
               [40., 20.,  0., 20., 40., 40., 20., 40., 20., 20., 20., 20., 20.,
                20.],
               [40., 60., 20., 40., 40., 20.,  0., 20., 60., 60., 40.,  0.,  0.,
                20.]])
        """
        if step_size is None:
            step_size = t_window # Non-overlapping time bins by default

        t = self.time
        t_min, t_max = np.min(t), np.max(t)

        # Ensure the time window is less than the total duration
        assert t_window < t_max - t_min
        
        # Calculate the number of steps
        n_step = int((t_max - t_min - t_window) // step_size) + 1
        neural_traj = np.zeros((self.activity.shape[0], n_step + 1))

        # Define the edges of the time bins
        left_bounds = np.linspace(t_min, t_min + step_size * n_step, n_step + 1)
        right_bounds = left_bounds + t_window

        for i in range(n_step + 1):
            neural_traj[:, i] = np.sum(
                self.activity[:, (t >= left_bounds[i])&(t < right_bounds[i])],
                axis= 1
            ) / t_window * 1000 # Convert to firing rate (Hz)

        return NeuralTrajectory(
            neural_trajectory=neural_traj, 
            time=left_bounds + t_window / 2

        )
    
    def calc_firing_rate(
        self, 
        nbins: tuple,
        is_remove_nan: bool = True,
        is_smoothing: bool = False,
        smooth_matrix: Optional[ndarray] = None,
        fps: float = 50,
        t_interv_limits: float = 100
    ) -> TuningCurve:
        """
        Calculate the firing rate of each neuron
        In calcium imaging, it is more appropriate to name it as `calcium
        event rate`, as spikes were putatively generated by deconvolving
        the calcium traces.
        
        Parameters
        ----------
        nbins: tuple
            The maximum number of bins in each dimension, e.g., (48, 48)
        is_remove_nan: bool
            Whether to remove the nan
        is_smoothing: bool
            Whether to smooth the firing rate
        smooth_matrix: ndarray
            The smoothing matrix
        fps: float
            The frame per second of recording
        t_interv_limits: float
            The maximum time interval allowed between two frames.
        
        Returns
        -------
        TuningCurve
            The firing rate of each neuron, with shape (n_neurons, nbins).
            
        Notes
        -----
        The firing rate is calculated by the following formula:
            firing_rate = Spike counts / occupation time (ms)
            
        Examples
        --------
        >>> from mazepy.datastruc.variables import SpikeTrain
        >>> from mazepy.datastruc.variables import VariableBin
        >>> import numpy as np
        >>>
        >>> spike = SpikeTrain(
        ...     activity = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0]]),
        ...     time = np.array([0, 50, 100, 150, 200, 250]),
        ...     variable = VariableBin(np.array([1, 1, 2, 2, 2, 2])),
        ... )
        >>> firing_rate = spike.calc_firing_rate(nbins = (2, ), fps = 20)
        >>> spike.firing_rate
        array([[20.,  5.],
               [ 0., 15.]])
        """
        _nbins = np.prod(nbins)
        dt = np.append(np.ediff1d(self.time), 1000 / fps)
        dt[dt > t_interv_limits] = t_interv_limits
        
        occu_time, _, _ = binned_statistic(
            self.variable,
            dt,
            bins=_nbins,
            statistic="sum",
            range = [0, _nbins + 0.00001]
        )
        self.occu_time = occu_time
        
        spike_count = np.zeros((self.activity.shape[0], _nbins), np.float64)
        for i in range(_nbins):
            idx = np.where(self.variable == i+1)[0]
            spike_count[:,i] = np.nansum(self.activity[:, idx], axis = 1)

        self.firing_rate = spike_count/(occu_time/1000)
        
        # Deal with nan value
        if is_remove_nan:
            self.firing_rate[np.isnan(self.firing_rate)] = 0
        
        self.firing_rate = self.firing_rate.astype(np.float64)
        
        if is_smoothing:
            assert smooth_matrix is not None
        
            self.firing_rate = np.dot(self.firing_rate, smooth_matrix)
            return TuningCurve(self.firing_rate, nbins)
        else:
            return TuningCurve(self.firing_rate, nbins)

class CalciumTraces(_NeuralActivity):
    """
    A class for representing calcium traces.
    """
    def __init__(
        self,
        activity: ndarray,
        time: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Parameters
        ----------
        activity: ndarray
            The calcium traces of each neuron, with shape (n_neurons, n_time).
        time: Variable1D or ndarray
            The time stamp of each time point, with shape (n_time, ).
        variable: VariableBin
            The bin index of each time point, with shape (n_time, ).
        """
        super().__init__(activity, time, variable)
        
    def binarize(
        self, 
        thre: float = 3., 
        func: Callable = np.nanstd
    ) -> SpikeTrain:
        """Binarize deconvolved signal to calcium events (putative spikes).

        Parameters
        ----------
        thre : float, optional
            The threshold to identify events, by default 3.
        func : Callable, optional
            The function to calculate the basic level for binarization, 
            by default np.std

        Returns
        -------
        Calcium events : SpikeTrain
            The binarized deconvolved signal of each neuron, with shape 
            (n_neurons, n_time).
            
        Examples
        --------
        >>> from mazepy.datastruc.variables import CalciumTraces
        >>> from mazepy.datastruc.variables import VariableBin
        >>> import numpy as np
        >>>
        >>> calcium = CalciumTraces(
        ...     activity = np.array([
        ...         [0.02, 0.03, 0.01, 2, 0.1, 0.05, 
        ...          0.01, 0., 0.003, 0.03, 0.02, 0.03],
        ...         [0.06, 3., 0.05, 0.08, 0.03, 0.05, 
        ...          0.02, 0.01, 0.06, 0.03, 0.02, 0.03]
        ...     ]),
        ...     time = np.array([
        ...         0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550
        ...     ]),
        ...     variable = VariableBin(np.array([
        ...         1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1
        ...     ]))
        ... )
        >>> spike = calcium.binarize(thre = 3)
        >>> spike.activity
        array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
               
        i.e. the two neurons.
        """
        try:
            thre_values = np.reshape(thre * func(self.activity, axis = 1),
                                     [self.activity.shape[0], 1])
        except:
            raise ValueError("The input function should have 'axis' methods.")
        
        return SpikeTrain(
            activity = np.where(self.activity - thre_values >= 0, 1, 0),
            time = self.time,
            variable = VariableBin(self.variable)
        )
        
class RawSpikeTrain(_NeuralActivity):
    """
    A class for representing raw spike trains.
    """
    def __init__(
        self,
        activity: ndarray,
        time: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Parameters
        ----------
        activity: ndarray
            The spike trains of each neuron, with shape (n_neurons, n_time).
        time: Variable1D
            The time stamp of each time point, with shape (n_time, ).
        variable: VariableBin
            The bin index of each time point, with shape (n_time, ).
        """
        super().__init__(activity, time, variable)

class KilosortSpikeTrain(SpikeTrain):
    """
    A class for the spike train generated by KiloSort, which is a neural data
    spike sorting algorithm. This class processes raw spike identification
    data into a structured spike train array.
    """
    def __init__(
        self,
        activity: ndarray,
        time: ndarray,
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Initializes a processed spike train from raw Kilosort output.

        Parameters
        ----------
        activity : ndarray
            The neuron ID of each spike, with shape (n_time, ).
        time : ndarray. Unit: miliseconds
            The time stamp of each time point, with shape (n_time, ).
        variable : Optional[VariableBin]
            The bin index of each time point, with shape (n_time, ), optional.
        """
        # Ensure the time is sorted and sort activity and variable 
        # accordingly
        sort_idx = np.argsort(time)
        sorted_activity = activity[sort_idx]
        sorted_time = Variable1D(time[sort_idx], meaning='time')
        
        if variable is not None:
            if isinstance(variable, VariableBin):
                variable.bins = variable.bins[sort_idx]
            else:
                raise TypeError(
                    f"Expected variable of type VariableBin, "
                    f"got {type(variable)} instead."
                )
        
        # Convert neuron IDs to a binary spike train matrix
        spike_train = self._process(activity=sorted_activity)
        
        # Initialize the base SpikeTrain class with sorted and processed data
        super().__init__(
            activity=spike_train,
            time=sorted_time,
            variable=variable
        )

    def _process(self, activity: ndarray) -> ndarray:
        """
        Converts neuron IDs in 'activity' to a binary spike train matrix.

        Parameters
        ----------
        activity : ndarray
            Array of neuron IDs for each spike.

        Returns
        -------
        ndarray
            A binary matrix (n_neurons x n_time) indicating spike occurrences.
        """
        # Find the number of neurons based on the highest neuron ID
        n_neuron = np.max(activity)
        spike_train = np.zeros((n_neuron, activity.size), dtype=np.int64)
        
        # Populate the spike train matrix
        for neuron_id in range(1, n_neuron + 1):
            spike_train[neuron_id - 1, activity == neuron_id] = 1
        
        return spike_train


if __name__ == "__main__":
    import doctest
    doctest.testmod()