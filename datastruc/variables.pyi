from numpy import ndarray
from typing import Optional, Union, Callable
import numpy as np

class VariableBin:
    bins: ndarray

    def __init__(self, bins: ndarray) -> None: ...

class Variable1D:
    x: ndarray
    meaning: Optional[str]
    bins: Optional[VariableBin]

    def __init__(self, x: ndarray) -> None: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def transform_to_bin(
        self, 
        xbin: int, 
        xmax: float, 
        xmin: Optional[float] = ...
    ) -> VariableBin: ...

class Variable2D:
    x: ndarray
    y: ndarray
    meaning: Optional[str]
    bins: Optional[VariableBin]

    def __init__(self, x: ndarray, y: ndarray) -> None: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def transform_to_bin(
        self, 
        xbin: int, 
        ybin: int, 
        xmax: float, 
        ymax: float, 
        xmin: Optional[float] = ..., 
        ymin: Optional[float] = ...
    ) -> VariableBin: ...

class Variable3D:
    x: ndarray
    y: ndarray
    z: ndarray
    meaning: Optional[str]
    bins: Optional[VariableBin]

    def __init__(
        self, 
        x: ndarray, 
        y: ndarray, 
        z: ndarray, 
        meaning: Optional[str]
    ) -> None: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    def transform_to_bin(
        self, xbin: int, 
        ybin: int, 
        zbin: int, 
        xmax: float, 
        ymax: float, 
        zmax: float,
        xmin: Optional[float] = ..., 
        ymin: Optional[float] = ..., 
        zmin: Optional[float] = ...
    ) -> VariableBin: ...

class TuningCurve:
    _ndim: int
    firing_rate: ndarray
    nbins: int

    def __init__(self, firing_rate: ndarray) -> None: ...
    @property
    def n_neuron(self) -> int: ...

class _NeuralActivity:
    variable: Optional[VariableBin]
    activity: ndarray
    time_stamp: ndarray

    def __init__(
        self, 
        activity: ndarray, 
        time_stamp: Variable1D, 
        variable: Optional[VariableBin],
        ctype: Optional[str] = ...
    ) -> None: ...
    def __str__(self) -> str: ...
    def __len__(self) -> int: ...
    @property
    def shape(self) -> tuple: ...
    @property
    def n_neuron(self) -> int: ...
    def remove_nan(self) -> None: ...

class CalciumTraces(_NeuralActivity):
    def __init__(
        self,
        activity: ndarray,
        time_stamp: Variable1D,
        variable: Optional[VariableBin]
    ) -> None: ...
    def binarize(
        self, 
        thre: float = 3., 
        func: Callable = np.nanstd
    ) -> SpikeTrain: ...

class RawSpikeTrain(_NeuralActivity):
    def __init__(
        self,
        activity: ndarray,
        time_stamp: Variable1D,
        variable: Optional[VariableBin]
    ) -> None: ...

class SpikeTrain(_NeuralActivity):
    variable: Optional[VariableBin]
    activity: ndarray
    time_stamp: ndarray
    firing_rate: Optional[ndarray]
    occu_time: Optional[ndarray]

    def __init__(
        self,
        activity: ndarray,
        time_stamp: Variable1D,
        variable: Optional[VariableBin]
    ) -> None: ...
    def calc_temporal_tuning_curve(self, twindow: float) -> TuningCurve: ...
    def calc_firing_rate(
        self, 
        nbins: tuple,
        is_remove_nan: bool = True,
        is_smoothing: bool = False,
        smooth_matrix: Optional[ndarray] = None,
        fps: float = 50,
        t_interv_limits: float = 100
    ) -> TuningCurve: ...

class KilosortSpikeTrain(SpikeTrain):
    variable: Optional[VariableBin]
    activity: ndarray
    time_stamp: ndarray
    firing_rate: Optional[ndarray]
    occu_time: Optional[ndarray]
    
    def __init__(
        self, 
        activity: ndarray, 
        time_stamp: Variable1D, 
        variable: VariableBin | None
    ) -> None: ...
    def _process(self, activity: np.ndarray) -> np.ndarray: ...