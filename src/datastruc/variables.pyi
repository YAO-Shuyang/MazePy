from numpy import ndarray
from typing import Optional, Union, Callable
import numpy as np

class VariableBin(ndarray):
    def __new__(cls, input_array: ndarray) -> 'VariableBin': ...
    def __array_finalize__(self, obj: Optional[ndarray]) -> None: ...

class Variable1D(ndarray):
    meaning: Optional[str]

    def __new__(
        cls, 
        input_array: ndarray, 
        meaning: Optional[str] = ...
    ) -> 'Variable1D': ...
    def __array_finalize__(self, obj: Optional[ndarray]) -> None: ...
    @property
    def x(self) -> ndarray: ...
    def to_bin(
        self, 
        xmin: float,
        xmax: float,
        nbin: int
    ) -> VariableBin: ...

class Variable2D(ndarray):
    meaning: Optional[str]

    def __new__(
        cls, 
        input_array: ndarray,
        meaning: Optional[str] = ...
    ) -> 'Variable2D': ...
    def __array_finalize__(self, obj: Optional[ndarray]) -> None: ...
    @property
    def x(self) -> Variable1D: ...
    @property
    def y(self) -> Variable1D: ...
    def to_bin(
        self, 
        xmin: float,
        xmax: float,
        xnbin: int,
        ymin: float,
        ymax: float,
        ynbin: int
    ) -> VariableBin: ...

class Variable3D(ndarray):
    meaning: Union[str, tuple[str, str, str], None] = None

    def __new__(
        cls, 
        input_array: ndarray,
        meaning: Union[str, tuple[str, str, str], None] = ...
    ) -> 'Variable3D': ...
    def __array_finalize__(self, obj: Optional[ndarray]) -> None: ...
    @property
    def x(self) -> Variable1D: ...
    @property
    def y(self) -> Variable1D: ...
    @property
    def z(self) -> Variable1D: ...
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
    ) -> VariableBin: ...

class TuningCurve(ndarray):
    def __init__(self, firing_rate: ndarray) -> None: ...
    @property
    def n_neuron(self) -> int: ...
    def get_argpeaks(self) -> ndarray: ...
    def get_peaks(self) -> ndarray: ...
    def remove_nan(self) -> None: ...
    def get_fields(self) -> list[dict]: ...
    def 

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