"""
Grid Basic and GridSize defines the basic class of an environment.
"""

class GridBasic(object):
    '''
    Parameter
    ---------
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    '''
    def __init__(self, xbin: int, ybin: int) -> None:
        self._xbin, self._ybin = None, None
        self._xbin = xbin
        self._ybin = ybin

    @property
    def xbin(self):
        return self._xbin
    
    @xbin.setter
    def xbin(self, xbin_val: int):
        if type(xbin_val) is int and xbin_val >= 1:
            self._xbin = xbin_val
        else:
            raise ValueError(f"xbin should get a int value larger than 1, but it gets {xbin_val} instead.")
    
    @property
    def ybin(self):
        return self._ybin
    
    @ybin.setter
    def ybin(self, ybin_val: int):
        if type(ybin_val) is int and ybin_val >= 1:
            self._ybin = ybin_val
        else:
            raise ValueError(f"ybin should get a int value larger than 1, but it gets {ybin_val} instead.")



class GridSize(GridBasic):
    '''
    Note
    ----
    The basic parameter to bin the value space into grid-like bins.

    Parameter
    ---------
    xmax: float, required
        The top limit of recorded data on dimension x.
    ymax: float, required
        The top limit of recorded data on dimension y.
    xmin: float, required
        The bottom limit of recorded data on dimension x.
    ymin: float, required
        The bottom limit of recorded data on dimension y.
    '''
    def __init__(self, 
                 xbin: int, 
                 ybin: int, 
                 xmax: float, 
                 ymax: float, 
                 xmin: int or float = 0, 
                 ymin: int or float = 0
                ) -> None:
        super().__init__(xbin, ybin)
        self._xmax = xmax
        self._ymax = ymax
        self._xmin = xmin
        self._ymin = ymin

        if xmin >= xmax:
            raise ValueError(f"Detect value xmin({xmin}) >= xmax({xmax}). However, xmin should not be equal to or bigger than xmax.")

        if ymin >= ymax:
            raise ValueError(f"Detect value ymin({ymin}) >= ymax({ymax}). However, ymin should not be equal to or bigger than ymax.")
    

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, value):
        if value > self.xmin:
            self._xmax = value
        else:
            raise ValueError(f"input xmax ({value}) should be bigger than xmin {self.xmin}, but it is not.")
    
    @property
    def ymax(self):
        return self._ymax

    @ymax.setter
    def ymax(self, value):
        if value > self.ymin:
            self._ymax = value
        else:
            raise ValueError(f"input ymax ({value}) should be bigger than ymin {self.ymin}, but it is not.")
    
    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, value):
        if value < self.xmax:
            self._xmin = value
        else:
            raise ValueError(f"input xmin ({value}) should be smaller than {self.xmax}, but it is not.")
    
    @property
    def ymin(self):
        return self._ymin

    @ymin.setter
    def ymin(self, value):
        if value < self.ymax:
            self._ymin = value
        else:
            raise ValueError(f"input ymin ({value}) should be smaller than {self.ymax}, but it is not.")