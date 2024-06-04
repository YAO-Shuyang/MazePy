import numpy as np

ShapeLikeStructure = tuple | list | np.ndarray

class RateMap(object):
    def __init__(self, rate_map: np.ndarray) -> None:
        self._value = rate_map
        
    @property
    def value(self):
        return self._value
    
    def reshape(self, newshape: ShapeLikeStructure):
        return np.reshape(self._value, newshape)
    
    @property
    def peak_rate(self):
        return np.nanmax(self._value)
    
    @property
    def lowest_rate(self):
        return np.nanmin(self._value)
    
    """
    In future, add smooth functions.
    """