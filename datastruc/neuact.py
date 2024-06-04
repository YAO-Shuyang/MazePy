from numpy import ndarray
import numpy as np
import time
import warnings
from typing import Optional, Union, Callable
from scipy.stats import binned_statistic

from mazepy.datastruc.kernel import GaussianKernel1d, UniformKernel1d
from mazepy.datastruc.exceptions import ViolatedConstraintError
from mazepy.datastruc.exceptions import DimensionMismatchError
from mazepy.datastruc.variables import VariableBin, Variable1D
from mazepy.basic.conversion import coordinate_recording_time
from mazepy.basic._calc_rate import _get_spike_counts, _get_occu_time

Kernels = Union[GaussianKernel1d, UniformKernel1d, ndarray]

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
    >>> from mazepy.datastruc.neuact import TuningCurve
    >>> firing_rates = np.array([[0, 2, 3], [1, 5, 2], [3, 0, 0]])
    >>> occu_time = np.array([50, 100, 150])
    >>> tuning_curve = TuningCurve(firing_rates, occu_time)
    >>> print(tuning_curve.n_neuron)
    3
    >>> print(tuning_curve.get_argpeaks())
    [2 1 0]
    >>> print(tuning_curve.get_peaks())
    [3 5 3]
    """
    occu_time: np.ndarray

    def __new__(
        cls, 
        firing_rate: ndarray, 
        occu_time: np.ndarray
    ) -> 'TuningCurve':
        obj = np.asarray(firing_rate).view(cls)
        return obj
    
    def __init__(
        self, 
        firing_rate: ndarray, 
        occu_time: np.ndarray
    ) -> None:
        self.occu_time = occu_time
    
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
        self.remove_nan()
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
    >>> import numpy as np
    >>> from mazepy.datastruc.neuact import NeuralTrajectory
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
        return obj
    
    def __init__(
        self, 
        neural_trajectory: ndarray, 
        time: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None
    ) -> None:
        if time.shape[0] != self.shape[1]:
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
        
        self.time = time
        self.variable = variable
        pass

    def __array_finalize__(self, obj: Optional[ndarray]) -> None:
        if obj is None:
            return
        self.time = getattr(obj, 'time', None)
        self.variable = getattr(obj, 'variable', None)
        
    @property
    def n_neuron(self) -> int:
        return self.shape[0]

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
        obj = np.asarray(activity).view(cls)
        return obj
    
    def __init__(
        self, 
        activity: ndarray, 
        time: ndarray, 
        variable: Optional[VariableBin] = None    
    ) -> None:

        if self.ndim == 1:
            self = self[np.newaxis, :]
        elif self.ndim > 2:
            raise DimensionMismatchError(
                len1=self.ndim, 
                len2=2,
                message="The shape of activity should be (n_neurons, n_time),"
            )    

        if self.shape[1] != time.shape[0]:
            raise DimensionMismatchError(
                len1=self.shape[1], 
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
            
        self.time = time
        self.variable = variable
            
        self.remove_nan()
    
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

    Methods
    -------
    calc_total_time(self, t_interv_limits: Optional[float] = None)
        Get the total time of the spike train.
    calc_spike_count(self)
        Calculate the spike count of each neuron.
    calc_mean_rate(self, t_interv_limits: Optional[float] = None)
        Calculate the mean firing rate of each neuron.


    Raises
    ------
    ViolatedConstraintError
        If the activity data are not binary.
    """
    def __new__(
        cls, 
        activity: ndarray, 
        time: Union[Variable1D, ndarray], 
        variable: Optional[VariableBin] = None
    ):
        obj = np.asarray(activity).view(cls)
        return obj
    
    def __init__(
        cls,
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

    def _get_dt(self, t_interv_limits: Optional[float] = None) -> ndarray:
        """
        Get the time interval between each spike in the spike train, namely
        the time interval of each bin.
        
        Parameters
        ----------
        t_interv_limits: float
            The maximum time interval of the spike train in miliseconds. If 
            not provided, the 2 folds of the median time interval of the spike 
            train will be used.        
        """
        dt = np.ediff1d(self.time)

        if t_interv_limits is None:
            t_interv_limits = np.nanmedian(dt) * 2
        dt = np.append(dt, t_interv_limits)
        dt[dt > t_interv_limits] = t_interv_limits    
        return dt
    
    def calc_total_time(self, t_interv_limits: Optional[float] = None) -> float:
        """
        Get the total time of the spike train.

        Returns
        -------
        float
            The total time of the spike train in miliseconds.
        """
        return np.sum(self._get_dt(t_interv_limits))

    def calc_occu_time(
        self, 
        t_interv_limits: Optional[float] = None,
        nbins: Optional[int] = None
    ) -> ndarray:
        """
        Calculate the total time animal spent on each variable bin. For instance, 
        if they dwell in the same bin for 30 seconds, the occupation time of that
        bin will be 30 seconds.
        
        Parameters
        ----------
        t_interv_limits: float
            The maximum time interval of the spike train in miliseconds. If 
            not provided, the 2 folds of the median time interval of the spike 
            train will be used.        
        nbins: int
            The number of bins to calculate the occupation time.

        Returns
        -------
        ndarray, shape (nbins, ) )
            The occupation time of each variable bin. 
        """
        dt = self._get_dt(t_interv_limits)


        if nbins is None:
            warnings.warn(
                "nbins is not provided. The max value of the variable"
                f" {np.max(self.variable)} will be used."
            )
            nbins = np.max(self.variable) + 1
        else:
            if nbins <= np.max(self.variable):
                raise OverflowError(
                    f"The maximum bin ID of variables, "
                    f"{np.max(self.variable)}, "
                    f"equals to or exceeds the input bin number {nbins}."
                )
        
        return _get_occu_time(
            self.variable.astype(np.int64), 
            dt.astype(np.float64), 
            nbins
        )     

    def calc_spike_count(
        self, 
        mode: str = 'sum', 
        nbins: Optional[int] = None
    ) -> ndarray:
        """
        Calculate the spike count of each neuron.

        Parameters
        ----------
        mode: str, by default 'sum'.
            The mode of the spike count. If 'sum', the sum of the spike counts
            will be returned. If 'bin', the spike counts of each bin will be
            returned.
        nbins: int, by default None
            The number of bins to calculate the spike count. Only used when
            mode is 'bin'.

        Returns
        -------
        ndarray
            The total spike count of each neuron in the spike train (n_neuron, ),
            or the spike count on each bin in the spike train (n_neuron, n_bin).
            
        Raises
        ------
        ValueError
            If the mode is not 'sum' or 'bin'.
        
        Examples
        --------
        >>> spike_train = SpikeTrain(
        ...     activity=np.array([
        ...         [0, 0, 0, 1, 1, 1, 1, 1],
        ...         [0, 1, 1, 0, 0, 1, 1, 1],
        ...         [1, 0, 0, 1, 0, 0, 1, 1],
        ...         [0, 1, 0, 0, 1, 0, 0, 1],
        ...     ]),
        ...     time=np.linspace(0, 350, 8),
        ...     variable=np.array([0, 0, 0, 0, 0, 1, 1, 1])
        ... )
        >>> spike_train.calc_spike_count(mode='sum')
        SpikeTrain([5, 5, 4, 3])
        >>> spike_train.calc_spike_count(mode='bin', nbins=2)
        array([[2, 3],
               [2, 3],
               [2, 2],
               [2, 1]], dtype=int64)
        """
        if mode not in ['sum', 'bin']:
            raise ValueError("mode should be either 'sum' or 'bin'.")
        
        if mode == 'sum':
            return np.sum(self, axis=1)
        elif mode == 'bin':
            if nbins is None:
                warnings.warn(
                    "nbins is not provided. The max value of the variable"
                    f" {np.max(self.variable)} will be used."
                )
                nbins = np.max(self.variable) + 1
            else:
                if nbins <= np.max(self.variable):
                    raise OverflowError(
                        f"The maximum bin ID of variables, "
                        f"{np.max(self.variable)}, "
                        f"equals to or exceeds the input bin number {nbins}."
                    )
            return _get_spike_counts(
                self.astype(np.int64), 
                self.variable.astype(np.int64), 
                nbins
            )
        else:
            raise ValueError("Param mode should be either 'sum' or 'bin'.")

    def calc_mean_rate(
        self, 
        t_interv_limits: Optional[float] = None
    ) -> ndarray:
        """
        Calculate the mean firing rate of each meuron.

        Parameters
        ----------
        t_interv_limits: float
            The maximum time interval of the spike train in miliseconds. If 
            not provided, the 2 folds of the median time interval of the spike 
            train will be used.

        Returns
        -------
        ndarray
            The mean firing rate (Hz) of each neuron in the spike train.
            
        Notes
        -----
        The mean firing rate is calculated as the number of spikes per second
        divided by the total time of the spike train.
        
        The spike count can be obtained by calling `calc_spike_count()`.
        The total time can be obtained by calling `calc_total_time()`.
        
        Examples
        --------
        >>> spike_train = SpikeTrain(
        ...     activity=np.array([
        ...         [0, 0, 0, 1, 1, 1, 1, 1],
        ...         [0, 1, 1, 0, 0, 1, 1, 1],
        ...         [1, 0, 0, 1, 0, 0, 1, 1],
        ...         [0, 1, 0, 0, 1, 0, 0, 1],
        ...     ]),
        ...     time=np.linspace(0, 350, 8),
        ...     variable=np.array([0, 0, 0, 0, 0, 1, 1, 1])
        ... )
        >>> spike_train.calc_mean_rate(t_interv_limits=50)
        SpikeTrain([12.5, 12.5, 10. ,  7.5])
        """
        return (
            self.calc_spike_count()/self.calc_total_time(t_interv_limits)*1000
        )

    def calc_variable_trajectory(
        self,
        traj_time: Union[Variable1D, np.ndarray]
    ) -> Optional[Variable1D]:
        """
        Calculate the variable trajectory related to neural trajectory, namely 
        a 1D variable with the same length as the neural trajectory.

        Only computed if the variable is accessible.

        Returns
        -------
        Optional[Variable1D]
            The variable trajectory related to the neural trajectory.
        """
        if self.variable is not None:
            idx = coordinate_recording_time(traj_time, self.time)
            return self.variable[idx]
        else:
            return None

    def calc_neural_trajectory(
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
        >>> from mazepy.datastruc.neuact import SpikeTrain
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
        >>> tuning_curve = spikes.calc_neural_trajectory(50)
        >>> # time bins: [0, 50], [50, 100], [100, 150], ...
        >>> tuning_curve
        NeuralTrajectory([[40., 20.,  0., 20., 40., 20.,  0.],
                          [40.,  0., 40., 40., 20.,  0., 20.],
                          [40., 40., 20., 20., 40.,  0., 20.]])
        >>> tuning_curve = spikes.calc_neural_trajectory(50, 20)
        >>> # time bins: [0, 50], [20, 70], [40, 90], [60, 110], ...
        >>> tuning_curve
        NeuralTrajectory([[40., 40., 20.,  0.,  0.,  0., 20., 20., 20., 20., 40.,
                           40., 40., 20.],
                          [40., 20.,  0., 20., 40., 40., 20., 40., 20., 20., 20.,
                           20., 20., 20.],
                          [40., 60., 20., 40., 40., 20.,  0., 20., 60., 60., 40.,
                            0.,  0., 20.]])
        """
        if step_size is None:
            step_size = t_window # Non-overlapping time bins by default

        t = self.time
        t_min, t_max = np.min(t), np.max(t)

        # Ensure the time window is less than the total duration
        assert t_window < t_max - t_min
        
        # Calculate the number of steps
        n_step = int((t_max - t_min - t_window) // step_size) + 1
        neural_traj = np.zeros((self.shape[0], n_step + 1))

        # Define the edges of the time bins
        left_bounds = np.linspace(t_min, t_min + step_size * n_step, n_step + 1)
        right_bounds = left_bounds + t_window

        for i in range(n_step + 1):
            neural_traj[:, i] = np.sum(
                self[:, (t >= left_bounds[i])&(t < right_bounds[i])],
                axis= 1
            ) / t_window * 1000 # Convert to firing rate (Hz)

        return NeuralTrajectory(
            neural_trajectory=neural_traj, 
            time=left_bounds + t_window / 2
        )
    
    def calc_tuning_curve(
        self, 
        nbins: int,
        is_remove_nan: bool = True,
        t_interv_limits: Optional[float] = None
    ) -> TuningCurve:
        """
        Calculate the firing rate of each neuron
        In calcium imaging, it is more appropriate to name it as `calcium
        event rate`, as spikes were putatively generated by deconvolving
        the calcium traces.
        
        Parameters
        ----------
        nbins: int
            The number of bins. If variables are multiple-dimensional,
            it should be the product of the number of bins in each dimension.
        is_remove_nan: bool
            Whether to remove the nan
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
        >>> from mazepy.datastruc.neuact import SpikeTrain
        >>> from mazepy.datastruc.variables import VariableBin, Variable1D
        >>> import numpy as np
        >>>
        >>> spike = SpikeTrain(
        ...     activity = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0]]),
        ...     time = Variable1D([0, 50, 100, 150, 200, 250]),
        ...     variable = VariableBin([0, 0, 1, 1, 1, 1]),
        ... )
        >>> firing_rate = spike.calc_tuning_curve(
        ...     nbins = 2,
        ...     t_interv_limits = 50
        ... )
        >>> firing_rate
        TuningCurve([[20.,  5.],
                     [ 0., 15.]])
        """
        if nbins <= np.max(self.variable):
            raise OverflowError(
                f"The maximum bin ID of variables, {np.max(self.variable)}, "
                f"equals to or exceeds the input bin number {nbins}."
            )
        
        t1 = time.time()
        dt = self._get_dt(t_interv_limits=t_interv_limits)
        
        occu_time, _, _ = binned_statistic(
            self.variable,
            dt,
            bins=nbins,
            statistic="sum",
            range = [0, nbins - 1 + 0.00001]
        )
        #print(time.time() - t1)
        t2 = time.time()
        occu_time2 = self.calc_occu_time(
            t_interv_limits=t_interv_limits,
            nbins=nbins
        )
        #print(time.time() - t2)
        
        t3 = time.time()
        spike_count = np.zeros((self.shape[0], nbins), np.float64)
        for i in range(nbins):
            idx = np.where(self.variable == i)[0]
            spike_count[:,i] = np.nansum(self[:, idx], axis = 1)

        #print(time.time() - t3)
        
        t4 = time.time()
        spike_count2 = self.calc_spike_count(mode="bin", nbins=nbins)
        #print(time.time() - t4)

        firing_rate = spike_count/(occu_time/1000)
        
        # Deal with nan value
        if is_remove_nan:
            firing_rate[np.isnan(firing_rate)] = 0
        
        firing_rate = firing_rate.astype(np.float64)

        return TuningCurve(firing_rate, occu_time)

class CalciumTraces(_NeuralActivity):
    """
    A class for representing calcium traces.
    """
    def __new__(
        cls,
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
        return super().__new__(cls, activity, time, variable)
    
    def __init__(
        self,
        activity: ndarray,
        time: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None
    ) -> None:
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
        >>> from mazepy.datastruc.neuact import CalciumTraces
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
        >>> spike
        SpikeTrain([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
               
        i.e. the two neurons.
        """
        try:
            thre_values = np.reshape(
                thre * func(self, axis = 1),
                [self.shape[0], 1]
            )
        except:
            raise ValueError("The input function should have 'axis' methods.")
        
        return SpikeTrain(
            activity = np.where(self - thre_values >= 0, 1, 0),
            time = self.time,
            variable = VariableBin(self.variable)
        )
        
class RawSpikeTrain(_NeuralActivity):
    """
    A class for representing raw spike trains.
    """
    def __new__(
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
        return super().__new__(
            activity = activity,
            time = time,
            variable = variable
        )
        
    def __init__(self) -> None:
        super().__init__()

class KilosortSpikeTrain(SpikeTrain):
    """
    A class for the spike train generated by KiloSort, which is a neural data
    spike sorting algorithm. This class processes raw spike identification
    data into a structured spike train array.
    """
    def __new__(
        cls,
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
                variable = variable[sort_idx]
            else:
                raise TypeError(
                    f"Expected variable of type VariableBin, "
                    f"got {type(variable)} instead."
                )
        
        # Convert neuron IDs to a binary spike train matrix
        obj = np.asarray(sorted_activity, dtype=np.int64).view(cls)
        spike_train = obj.process(activity=sorted_activity)

        # Initialize the base SpikeTrain class with sorted and processed data
        return super().__new__(
            cls,
            activity=spike_train,
            time=sorted_time,
            variable=variable
        )
        
    def __init__(
        self,
        activity: ndarray,
        time: ndarray,
        variable: Optional[VariableBin] = None
    ) -> None:
        super().__init__(activity, time, variable)

    def process(self, activity: ndarray) -> ndarray:
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