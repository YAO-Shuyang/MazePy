<<<<<<< HEAD
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
    
    @property
    def ymax(self):
        return self._ymax
    
    @property
    def xmin(self):
        return self._xmin
    
    @property
    def ymin(self):
=======
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
    
    @property
    def ymax(self):
        return self._ymax
    
    @property
    def xmin(self):
        return self._xmin
    
    @property
    def ymin(self):
>>>>>>> bebfe805e63d3226d6196e1b8ff4c78089a1de31
        return self._ymin