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
        twindow: float,
        mode: str = 'sliding'
    ) -> TuningCurve:
        """
        Calculate the temporal firing rate of neurons, namely the temporal
        population vector.
        
        Parameters
        ----------
        twindow: float
            The time window in miliseconds, e.g., 400
        mode: str
            The mode to compute the temporal tuning curve.
            Valid values: 'sliding', 'non-overlap'
            - 'sliding': a sliding window, for instance,\\
                [0, 400] -> [50, 450] -> [100, 500] -> [150, 550] -> ...
            - 'non-overlap': non-overlapping windows, for instance,\\
                [0, 400] -> [400, 800] -> [800, 1200] -> [1200, 1600] -> ...
            
        Returns
        -------
        TuningCurve
            The temporal population vector of each neuron.
            
        Notes
        -----
        The temporal population vector is calculated by:\\
        :math:`r(t)` = :math:`\sum_{i=1}^n r_i(t)`
            
        where :math:`r_i(t)` is the firing rate of neuron :math:`i` at time
        :math:`t`.
        """
        if mode == 'sliding':
            raise NotImplementedError
        elif mode == 'non-overlap':
            raise NotImplementedError
        else:
            raise NotImplementedError

    
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
        
        spike_count = np.zeros((self.activity.shape[0], _nbins), dtype = np.float64)
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
        ...     activity = np.array([[0.02, 0.03, 0.01, 2, 0.1, 0.05, 
        ...                           0.01, 0., 0.003, 0.03, 0.02, 0.03], 
        ...                          [0.06, 3., 0.05, 0.08, 0.03, 0.05, 
        ...                           0.02, 0.01, 0.06, 0.03, 0.02, 0.03]]),
        ...     time_stamp = np.array([0, 50, 100, 150, 200, 250,
        ...                            300, 350, 400, 450, 500, 550]),
        ...     variable = VariableBin(np.array([1, 1, 2, 2, 2, 2,
        ...                                      1, 2, 2, 1, 1, 1])),
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
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()