from numpy import ndarray
import numpy as np
from typing import Optional, Union
from mazepy.utils import value_to_bin

class VariableBin:
    """
    Binned variable class, e.g. time, 1-3D position, sensory gradient, etc.
    """
    def __init__(self, bins: ndarray) -> None:
        self.bins = bins

class Variable1D:
    """
    1D variable class, e.g. time, 1-D position, sensory gradient, etc.
    """
    def __init__(self, x: ndarray, meaning: Optional[str] = None) -> None:
        self.x = x
        self.meaning = meaning # The physical meaning of the variable
        self.bins = None
        
    def __str__(self) -> str:
        return f"The variable represents {self.meaning}, 
                 with length {len(self.x)}"
                 
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
        return f"The variable represents {self.meaning}, 
                 with length {len(self.x)}"

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
        return f"The variable represents {self.meaning}, 
                 with length {len(self.x)}"

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
        if self._ndim == 1:
            self.firing_rate = firing_rate
        else:
            self.firing_rate = np.reshape(
                firing_rate, 
                [firing_rate.shape[0]] + list(nbins)
            )
        self.bins = None

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
            
        
    
    @property
    def n_neuron(self) -> int:
        return self.firing_rate.shape[0]
    
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
            The neural activity of each neuron, with shape (n_neurons, n_time).
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
        time_stamp: Variable1D
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
        
    def calc_temporal_tuning_curve(self, twindow: float) -> TuningCurve:
        raise NotImplementedError
    
    def calc_firing_rate(
        self
    ) -> TuningCurve:
        """
        Calculate the firing rate of each neuron
        """
        self.firing_rate = self.activity.mean(axis=1)
        return TuningCurve(self.firing_rate)