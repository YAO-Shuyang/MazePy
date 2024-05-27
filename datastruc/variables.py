from numpy import ndarray
import numpy as np
from typing import Optional, Union, Callable

from mazepy.utils import value_to_bin
import warnings
from scipy.stats import binned_statistic

class VariableBin:
    """
    Represents a binned variable, such as time, position, or sensory gradients.
    """
    def __init__(self, bins: ndarray) -> None:
        self.bins = bins

class Variable1D:
    """
    Represents a one-dimensional variable, like time or a single spatial 
    dimension.
    """
    def __init__(self, x: ndarray, meaning: Optional[str] = None) -> None:
        self.x = x
        self.meaning = meaning # Description of the variable
        self.bins = None
        
    def __str__(self) -> str:
        return (
            f"The variable represents {self.meaning}, "
            f"with length {len(self.x)}"
        )
                 
    def __len__(self) -> int:
        return len(self.x)

    def transform_to_bin(
        self, 
        xbin: int, 
        xmax: float, 
        xmin: float = 0
    ) -> VariableBin:
        """
        Transform the value of the variable into bin index.
        """
        self.bins = VariableBin(value_to_bin(self.x, xbin, xmax, xmin))
        return self.bins
    
class Variable2D:
    """
    2D variable class, e.g. time, 2-D position, sensory gradient, etc.
    """
    def __init__(
        self, 
        x: ndarray, 
        y: ndarray, 
        meaning: Optional[str] = None
    ) -> None:
        # x and y must have the same length
        assert x.shape[0] == y.shape[0] 
        self.x = x
        self.y = y
        self.meaning = meaning # The physical meaning of the variable
        self.bins = None
        
    def __str__(self) -> str:
        return (
            f"The variable represents {self.meaning}, "
            f"with length {len(self.x)}"
        )

    def __len__(self) -> int:
        return len(self.x)

    def transform_to_bin(
        self, 
        xbin: int, 
        ybin: int, 
        xmax: float, 
        ymax: float, 
        xmin: float = 0, 
        ymin: float = 0
    ) -> VariableBin:
        """
        Transform the value of the variable into bin index.
        """
        xbins = value_to_bin(self.x, xbin, xmax, xmin)
        ybins = value_to_bin(self.y, ybin, ymax, ymin)
        self.bins = VariableBin(xbins + xbin*(ybins - 1))
        return self.bins
    
class Variable3D:
    """
    3D variable class, e.g. time, 3-D position, sensory gradient, or 
    multiplexed varaibles.
    """
    def __init__(
        self,
        x: ndarray, 
        y: ndarray, 
        z: ndarray, 
        meaning: Optional[str] = None
    ) -> None:
        # x, y, and z must have the same length
        assert x.shape[0] == y.shape[0] == z.shape[0]
        self.x = x
        self.y = y
        self.z = z
        self.meaning = meaning # The physical meaning of the variable
        self.bins = None
        
    def __str__(self) -> str:
        return (
            f"The variable represents {self.meaning}, "
            f"with length {len(self.x)}"
        )

    def __len__(self) -> int:
        return len(self.x)

    def transform_to_bin(
        self,
        xbin: int, 
        ybin: int, 
        zbin: int, 
        xmax: float, 
        ymax: float, 
        zmax: float, 
        xmin: float = 0, 
        ymin: float = 0, 
        zmin: float = 0
    ) -> VariableBin:
        """
        Transform the value of the variable into bin index.
        """
        xbins = value_to_bin(self.x, xbin, xmax, xmin)
        ybins = value_to_bin(self.y, ybin, ymax, ymin)
        zbins = value_to_bin(self.z, zbin, zmax, zmin)
        self.bins = VariableBin(xbins + xbin*(ybins - 1) + 
                                xbin*ybin*(zbins - 1))
        return self.bins
    
class TuningCurve:
    """
    A class for representing tuning curves.
    """
    def __init__(self, firing_rate: ndarray, nbins: tuple) -> None:
        """
        Parameters
        ----------
        firing_rate: ndarray
            The firing rate of each bin, with shape (n_neurons, nbins).
        nbins: tuple
            The maximum number of bins in each dimension, e.g., (48, 48)
        """
        self._ndim = len(nbins)
        self.firing_rate = firing_rate
        self.nbins = nbins

    @property
    def shape(self):
        return self.firing_rate.shape
    
    def __len__(self) -> int:
        return len(self.firing_rate)
    
    def __str__(self) -> str:
        return f"{self.firing_rate}"
    
    def __add__(self, other: 'TuningCurve') -> 'TuningCurve':
        if self._ndim != other._ndim:
            raise ValueError(
                f"The number of dimensions for summation should be the same, "
                f"rather than {self._ndim} and {other._ndim}."
            )
            
        return TuningCurve(self.firing_rate + other.firing_rate, self.shape)
    
    def __sub__(self, other: 'TuningCurve') -> 'TuningCurve':
        if self._ndim != other._ndim:
            raise ValueError(
                f"The number of dimensions for subtraction should be the same, "
                f"rather than {self._ndim} and {other._ndim}."
            )
            
        return TuningCurve(self.firing_rate - other.firing_rate, self.shape)
    
    def __mul__(self, other: float) -> 'TuningCurve':
        return TuningCurve(self.firing_rate * other, self.shape)
    
    def __truediv__(self, other: float) -> 'TuningCurve':
        if other == 0:
            warnings.warn("The divisor should not be 0.")
            
        return TuningCurve(self.firing_rate / other, self.shape)
    
    @property
    def n_neuron(self) -> int:
        return self.firing_rate.shape[0]
    
    def reshape(self) -> ndarray:
        """
        Reshape the firing rate to the same dimensions with variables.

        Returns
        -------
        ndarray
            The reshaped firing rate, with shape (n_neurons, nbins).
            
        Examples
        --------
        >>> tcurve = TuningCurve(
        ...     firing_rate = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        ...     nbins = (2, 2)
        ... )
        >>> tcurve.reshape().shape
        (2, 2, 2)
        >>> tcurve.firing_rate
        array([[[1, 2],
                [3, 4]],
        <BLANKLINE>
               [[5, 6],
                [7, 8]]])
        """
        if np.prod(self.nbins) != self.firing_rate.shape[1]:
            raise ValueError(
                f"The number of bins in each dimension should be the same, "
                f"rather than {self.nbins} and {self.firing_rate.shape[1]}."
            )
            
        if self._ndim > 1:
            self.firing_rate = np.reshape(
                self.firing_rate, 
                [self.firing_rate.shape[0]] + list(self.nbins)
            )
            
        return self.firing_rate
    
class _NeuralActivity:
    """
    A base class for representing neural activity.
    """
    def __init__(
        self, 
        activity: ndarray, 
        time_stamp: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Parameters
        ----------
        activity: ndarray
            The activity of each neuron, with shape (n_neurons, n_time).
        time_stamp: Variable1D
            The time stamp of each time point, with shape (n_time, ).
        variable: VariableBin
            The bin index of each time point, with shape (n_time, ).
        """
        if isinstance(variable, VariableBin) or variable is None:
            self.variable = variable
        else:
            raise TypeError(
                f"The type of variable should be VariableBin,"
                f"rather than {type(variable)}"
            )
        
        if isinstance(variable, VariableBin):
            assert activity.shape[1] == len(time_stamp) == len(variable.bins)
            self.variable = variable.bins
        else:
            assert activity.shape[1] == len(time_stamp)
        
        self.activity = activity
        
        if isinstance(time_stamp, Variable1D):
            self.time_stamp = time_stamp.x
        else:
            self.time_stamp = time_stamp
            
        self.remove_nan()
    
    def __str__(self) -> str:
        return f"Neuron number: {len(self.activity)}"

    def __len__(self) -> int:
        return len(self.activity)
    
    @property
    def shape(self) -> tuple:
        return self.activity.shape
    
    @property
    def n_neuron(self) -> int:
        return self.activity.shape[0]
    
    def remove_nan(self) -> None:
        """
        Remove the nan in the neural activity or variables
        """
        if self.variable is not None:
            nan_indices = np.where(
                (np.isnan(np.sum(self.activity, axis=0))) |
                (np.isnan(self.time_stamp)) |
                (np.isnan(self.variable))
            )[0]
            self.variable = np.delete(self.variable, nan_indices)
        else:
            nan_indices = np.where( 
                (np.isnan(np.sum(self.activity, axis=0))) |
                (np.isnan(self.time_stamp))
            )[0]
            
        self.activity = np.delete(self.activity, nan_indices, axis=1)
        self.time_stamp = np.delete(self.time_stamp, nan_indices)

class SpikeTrain(_NeuralActivity):
    """
    A class for representing spike trains.
    """
    def __init__(
        self,
        activity: ndarray,
        time_stamp: Variable1D,
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Initialize the SpikeTrain with activity data, time stamps, and 
        optional bin indices.

        Parameters
        ----------
        activity: ndarray
            The spike trains of each neuron, with shape (n_neurons, n_time).
            All entries should be either 1 or 0.
        time_stamp: Variable1D. Unit: miliseconds
            The time stamp of each time point, with shape (n_time, ).
        variable: VariableBin
            The bin index of each time point, with shape (n_time, ).
        """
        try:
            assert np.all((activity == 0) | (activity == 1))
        except:
            raise ValueError(
                "All entries of the spike train should be either 0 or 1. "
                "Please check the input data and try again."
            )
        super().__init__(activity, time_stamp, variable)
        self.firing_rate = None
        
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
        ...     time_stamp = np.array([
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

        t = self.time_stamp
        t_min, t_max = np.min(t), np.max(t)

        # Ensure the time window is less than the total duration
        assert t_window < t_max - t_min
        
        # Calculate the number of steps
        n_step = int((t_max - t_min - t_window) // step_size) + 1
        tuning_curve = np.zeros((self.activity.shape[0], n_step + 1))

        # Define the edges of the time bins
        left_bounds = np.linspace(t_min, t_min + step_size * n_step, n_step + 1)
        right_bounds = left_bounds + t_window

        for i in range(n_step + 1):
            tuning_curve[:, i] = np.sum(
                self.activity[:, (t >= left_bounds[i])&(t < right_bounds[i])],
                axis= 1
            ) / t_window * 1000 # Convert to firing rate (Hz)

        return TuningCurve(tuning_curve, (n_step + 1, ))
    
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
        ...     time_stamp = np.array([0, 50, 100, 150, 200, 250]),
        ...     variable = VariableBin(np.array([1, 1, 2, 2, 2, 2])),
        ... )
        >>> firing_rate = spike.calc_firing_rate(nbins = (2, ), fps = 20)
        >>> spike.firing_rate
        array([[20.,  5.],
               [ 0., 15.]])
        """
        _nbins = np.prod(nbins)
        dt = np.append(np.ediff1d(self.time_stamp), 1000 / fps)
        dt[dt > t_interv_limits] = t_interv_limits
        
        occu_time, _, _ = binned_statistic(
            self.variable,
            dt,
            bins=_nbins,
            statistic="sum",
            range = [0, _nbins + 0.00001]
        )
        
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
        time_stamp: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Parameters
        ----------
        activity: ndarray
            The calcium traces of each neuron, with shape (n_neurons, n_time).
        time_stamp: Variable1D or ndarray
            The time stamp of each time point, with shape (n_time, ).
        variable: VariableBin
            The bin index of each time point, with shape (n_time, ).
        """
        super().__init__(activity, time_stamp, variable)
        
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
        ...     time_stamp = np.array([
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
            time_stamp = self.time_stamp,
            variable = VariableBin(self.variable)
        )
        
class RawSpikeTrain(_NeuralActivity):
    """
    A class for representing raw spike trains.
    """
    def __init__(
        self,
        activity: ndarray,
        time_stamp: Union[Variable1D, ndarray],
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Parameters
        ----------
        activity: ndarray
            The spike trains of each neuron, with shape (n_neurons, n_time).
        time_stamp: Variable1D
            The time stamp of each time point, with shape (n_time, ).
        variable: VariableBin
            The bin index of each time point, with shape (n_time, ).
        """
        super().__init__(activity, time_stamp, variable)

class KilosortSpikeTrain(SpikeTrain):
    """
    A class for the spike train generated by KiloSort, which is a neural data
    spike sorting algorithm. This class processes raw spike identification
    data into a structured spike train array.
    """
    def __init__(
        self,
        activity: ndarray,
        time_stamp: ndarray,
        variable: Optional[VariableBin] = None
    ) -> None:
        """
        Initializes a processed spike train from raw Kilosort output.

        Parameters
        ----------
        activity : ndarray
            The neuron ID of each spike, with shape (n_time, ).
        time_stamp : ndarray
            The time stamp of each time point, with shape (n_time, ).
        variable : Optional[VariableBin]
            The bin index of each time point, with shape (n_time, ), optional.
        """
        # Ensure the time_stamp is sorted and sort activity and variable 
        # accordingly
        sort_idx = np.argsort(time_stamp)
        sorted_activity = activity[sort_idx]
        sorted_time_stamp = Variable1D(time_stamp[sort_idx], meaning='time')
        
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
            time_stamp=sorted_time_stamp,
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