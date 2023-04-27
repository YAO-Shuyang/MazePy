<<<<<<< HEAD
'''
Noted on March 14, 2023

This file is to perform transformation between ID of bins and location.

Here we defined these 3 terms in order to definitely illustrate what these functions aim 
  at doing.

xbins represents the number of bins on dimension x, while ybins represents the number of 
  bins on dimension y.

1. Bin Index (termed as idx): 
    The absolute Index of certain bin. If we define the value space into (xbin, ybin), 
    that the bin ID must belong to {1, 2, ..., xbin*ybin}, something like:

      M+1    M+2    ...   xbin*ybin 
       .      .     ...       .
       .      .     ...       .
       .      .     ...       .
    xbin+1  xbin+2  ...    xbin*2
       1      2     ...     xbin

2. Bin coordinate (abbreviated as loc):
    The x, y coordinate of where a certain bin locates at.

  (0, ybin-1) (1, ybin-1) ... (xbin-1, ybin-1)
       .           .      ...        .
       .           .      ...        .
       .           .      ...        .   
    (0, 1)      (1, 1)    ...   (xbin-1, 1)
    (0, 0)      (1, 0)    ...   (xbin-1, 0)

3. Precise value coordinate (termed as PVL/pvl)
    The precise recorded data (usually 2 dimension)

    For example, if the data is 2D spatial coordinate, it might be (10.843 cm, 85.492 cm)

    If the data is a kind of conjunctively paired data, such as data that jointly combined 
      voice frequencies and 1D spatial coordinate together, it might be (43.154 cm, 5185 Hz)

    It represents precise value of recorded data.


Function developed here are generally for transformation between these 3 forms of data.

Note that the transformation between loc and idx are reversible, but the transformation 
  from pvl to loc/idx are irreversible (for this kind of transformaiton will lose pretty 
  much precise information about the value location)

For this reason, only 4 function: 
                idx_to_loc, loc_to_idx, pvl_to_loc, pvl_to_idx 
  are available.
'''

import numpy as np
import warnings

from mazepy.behav.grid import GridSize, GridBasic


"""
Check whether a bin locates at the border of a maze.
"""
def isNorthBorder(BinID, xbin = 12, ybin = 12):
    if (BinID-1) // xbin == ybin - 1:
        return True
    else:
        return False

def isEastBorder(BinID, xbin = 12):
    if BinID % xbin == 0:
        return True
    else:
        return False

def isWestBorder(BinID, xbin = 12):
    if BinID % xbin == 1:
        return True
    else:
        return False

def isSouthBorder(BinID, xbin = 12):
    if BinID <= xbin:
        return True
    else:
        return False


"""
Define BinID class, BinCoordinate class, RawCoordinate class, NormCoordinate class
"""

class BinIDArray(GridBasic):
    def __init__(self, 
                 input_val: int or list or tuple or np.ndarray,
                 xbin: int, 
                 ybin: int
                ) -> None:
        super().__init__(xbin=xbin, ybin=ybin)
        self._values = None

        try:
            l = len(input)
            if type(input_val) is list or type(input_val) is tuple:
                self.values = np.array(input_val, dtype = np.int64)
            elif type(input_val) is np.ndarray:
                self.values = input_val.astype(np.int64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
        except:
            self.values = input_val

    @property
    def values(self):
        return self._value
    
    @values.setter
    def values(self, val):
        if type(val) is int:
            if val > self.xbin*self.ybin or val < 1:
                raise ValueError(f"{val} contains invalid value(s). Valid values belong to list [1, ..., {self.xbin*self.ybin}]")
            else:
                self._value = val
        else:
            if len(np.where((val > self.xbin*self.ybin)|(val < 1))[0]) != 0:
                raise ValueError(f"{val} contains invalid value(s). Valid values belong to list [1, ..., {self.xbin*self.ybin}]")
            else:
                self._value = val

    def __repr__(self):
        return 'BinIDArray ({}) at GridBasic ({} {})'.format(self._value, self.xbin, self.ybin)
        
    def __eq__(self, others) -> bool:
        return self._value == others.value
    
    def __ne__(self, others) -> bool:
        return self._value != others.value
    
    def __lt__(self, other) -> bool:
        return self._value < other.value
    
    def __gt__(self, other) -> bool:
        return self._value > other.value
    
    def to_binxy(self):
        return (self.values - 1) % self.xbin, (self.values - 1) // self.xbin

class BinCoordinateArray(GridBasic):
    def __init__(self, 
                 x: float or np.ndarray,
                 y: float or np.ndarray,
                 xbin: int, 
                 ybin: int
                ) -> None:
        super().__init__(xbin, ybin)
        self._x, self._y = None, None
        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, input_val: int or float or list or tuple or np.ndarray):
        try:
            l = len(input_val)
            if type(input_val) is list or type(input_val) is tuple:
                x = np.array(input_val, dtype = np.int64)
            elif type(input_val) is np.ndarray:
                x = input_val.astype(np.int64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((x > self.xbin-1)|(x < 0))[0]) != 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xbin-1}]")
                return
            else:
                self._x = x
        except:
            x = input_val
            if x > self.xbin-1 or x < 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xbin-1}]")
                return
            else:
                self._x = x
                
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, input_val):
        try:
            l = len(input_val)
            if type(input_val) is list or type(input_val) is tuple:
                y = np.array(input_val, dtype = np.int64)
            elif type(input_val) is np.ndarray:
                y = input_val.astype(np.int64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((y > self.ybin-1)|(y < 0))[0]) != 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ybin-1}]")
                return
            else:
                self._y = y
        except:
            y = input_val
            if y > self.ybin-1 or y < 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ybin-1}]")
                return
            else:
                self._y = y
        
    def to_binid(self):
        return self.x + self.y*self.ybin + 1
        

class RawCoordinateArray(GridSize):
    def __init__(self, 
                 x: float or np.ndarray,
                 y: float or np.ndarray,
                 xbin: int, 
                 ybin: int, 
                 xmax: float, 
                 ymax: float, 
                 xmin: int or float = 0, 
                 ymin: int or float = 0
                ) -> None:
        super().__init__(xbin, ybin, xmax, ymax, xmin, ymin)
        self._x, self._y = None, None
        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, input_val: int or float or tuple or np.ndarray):
        try:
            l = len(input_val)
            if type(input_val) is list or type(input_val) is tuple:
                x = np.array(input_val, dtype = np.float64)
            elif type(input_val) is np.ndarray:
                x = input_val.astype(np.float64)
            else:
                warnings.warn(f"Only int, float, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((x >= self.xmax)|(x < 0))[0]) != 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xmax})")
                return
            else:
                self._x = x

        except:
            x = input_val
            if x >= self.xmax or x < 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xmax})")
                return
            else:
                self._x = x          
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, input_val):
        try:
            l = len(input_val)
            if type(input_val) is list or type(input_val) is tuple:
                y = np.array(input_val, dtype = np.float64)
            elif type(input_val) is np.ndarray:
                y = input_val.astype(np.float64)
            else:
                warnings.warn(f"Only int, float, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((y >= self.ymax)|(y < 0))[0]) != 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ymax})")
                return
            else:
                self._y = y

        except:
            y = input_val
            if y >= self.ymax or y < 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ymax})")
                return
            else:
                self._y = y
    
    @property
    def Norm(self):
        return NormCoordinateArray((self.x - self.xmin) / (self.xmax + 0.0001 - self.xmin) * self.xbin, 
                                   (self.y - self.ymin) / (self.ymax + 0.0001 - self.ymin) * self.ybin, 
                                   xbin = self.xbin, ybin = self.ybin)
    
    def to_binid(self):
        return self.Norm.to_binid()
    
    def to_binxy(self):
        return self.Norm.to_binxy()
    

class NormCoordinateArray(GridBasic):
    def __init__(self, 
                 x: float or np.ndarray,
                 y: float or np.ndarray,
                 xbin: int, 
                 ybin: int
                ) -> None:
        super().__init__(xbin, ybin)
        self._x, self._y = None, None
        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, input_val: int or float or tuple or np.ndarray):
        try:
            l = len(input)
            if type(input_val) is list or type(input_val) is tuple:
                x = np.array(input_val, dtype = np.float64)
            elif type(input_val) is np.ndarray:
                x = input_val.astype(np.float64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((x >= self.xbin)|(x < 0))[0]) != 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xbin})")
                return
            else:
                self._x = x

        except:
            x = input_val
            if x >= self.xbin or x < 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xbin})")
                return
            else:
                self._x = x
                return


    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, input_val):
        try:
            l = len(input)
            if type(input_val) is list or type(input_val) is tuple:
                y = np.array(input_val, dtype = np.float64)
            elif type(input_val) is np.ndarray:
                y = input_val.astype(np.float64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((y >= self.ybin)|(y < 0))[0]) != 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ybin})")
                return
            else:
                self._y = y

        except:
            y = input_val
            if y >= self.ybin or y < 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ybin})")
                return
            else:
                self._y = y
                return
        
    def to_binid(self) -> np.ndarray:
        if type(self.x) is int or type(self.x) is float:
            return int(self.x) + int(self.y)*self.xbin + 1
        else:
            return ((self.x // 1) + (self.y // 1)*self.xbin + 1).astype(np.int64)
    
    def to_binxy(self) -> np.ndarray:
        if type(self.x) is int or type(self.x) is float:
            return int(self.x), int(self.y)
        else:
            return (self.x // 1).astype(np.int64), (self.y // 1).astype(np.int64)
        




# =============================================================================================== 
# =============================================================================================== 
# =============================================================================================== 
# =============================================================================================== 




def pvl_to_idx(x: int or float or tuple or np.ndarray, 
               y: int or float or tuple or np.ndarray, 
               xbin: int, 
               ybin: int,
               xmax: float or int, 
               ymax: float or int, 
               xmin: float or int = 0,
               ymin: float or int = 0
              ) -> np.ndarray:
    """
    Parameter
    ---------
    x, y: int or float or tuple or numpy.ndarray object, required.
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xmin: float, optional, default: 0
    ymin: float, optional, default: 0

    Return
    ------
    idx: int or numpy.ndarray 1d vector, containing index/indices of a collection of bin(s)

    Note
    ----
    Transform precise value location into bin index.
    """
    Arr = RawCoordinateArray(x, y, xbin=xbin, ybin=ybin, xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin)
    return Arr.to_binid()


# transform 960cm position data into nx*nx form
def pvl_to_loc(x: int or float or tuple or np.ndarray, 
               y: int or float or tuple or np.ndarray, 
               xbin: int,
               ybin: int,
               xmax: float or int, 
               ymax: float or int, 
               xmin: float or int = 0,
               ymin: float or int = 0
              ) -> np.ndarray:
    """
    Parameter
    ---------
    x, y: int or float or tuple or numpy.ndarray object, required.
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xmin: float, optional, default: 0
    ymin: float, optional, default: 0

    Return
    ------
    coordinate: int or numpy.ndarray object with a shape of (2, T), containing bin location(s) of a collection of bin(s)

    Note
    ----
    Transform precise value location into bin location.
    """
    Arr = RawCoordinateArray(x, y, xbin=xbin, ybin=ybin, xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin)
    return Arr.to_binxy()

def loc_to_idx(x: np.ndarray or int, y: np.ndarray or int, xbin: int, ybin: int) -> np.ndarray:
    '''
    Parameter
    ---------
    x, y: The bin coordinate.
    xbin, ybin: int, required
        The total bin number of dimension xcoordinate

    Return
    ------
    idx: int or numpy.ndarray 1d vector, the bin index

    Note
    ----
    Transform bin location into bin index.
    '''
    Arr = BinCoordinateArray(x, y, xbin=xbin, ybin=ybin)
    return Arr.to_binid()


def idx_to_loc(idx: int or np.ndarray, xbin: int, ybin: int) -> np.ndarray:
    '''
    Parameter
    ---------
    idx: int or np.ndarray 1d vector, required
        the bin index
    xbin, ybin: int, optional
        The total bin number of dimension x
        default: 10

    Return
    ------
    x, y: int or np.ndarray 1d vector, respectively. 
        that is the bin coordinate (x, y)

    Note
    ----
    Transform bin index into bin location. 
    '''
    Arr = BinIDArray(idx, xbin=xbin, ybin=ybin)
    return Arr.to_binxy()


def loc_to_edge(x: float, y: float, xbin: int, ybin: int) -> tuple or None:
    if (type(x) is float or type(x) is int) and (type(y) is float or type(y) is int):
        if x == int(x) and y != int(y):
            return ('v', int(x), int(y))
        elif x != int(x) and y == int(y):
            return ('h', int(x), int(y))
        else:
            return
    else:
        return


def nearest_edge(x: int or float or tuple or np.ndarray, 
               y: int or float or tuple or np.ndarray, 
               xbin: int,
               ybin: int,
               xmax: float or int, 
               ymax: float or int, 
               xmin: float or int = 0,
               ymin: float or int = 0
              ) -> tuple:
    '''
    Parameter
    ---------
    x, y: int or float or tuple or numpy.ndarray object, required.
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xmin: float, optional, default: 0
    ymin: float, optional, default: 0

    Return
    ------
    tuple: (direction, x, y) of the edge

    Note
    ----
    Find the nearest edge to a precise value point.
    '''
    Arr = RawCoordinateArray(x, y, xbin=xbin, ybin=ybin, xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin)
    xp, yp = Arr.Norm.x, Arr.Norm.y
    x, y = int(xp), int(yp)

    dy = yp - y - 0.5
    dx = xp - x - 0.5
    
    if np.abs(dx) <= np.abs(dy):
        if dy >= 0:
            return ('h', x, y+1)  # North
        else: # dy < 0
            return ('h', x, y)  # South
    else: # np.abs(dx) > np.abs(dy):
        if dx >= 0:  
            return ('v', x+1, y) # East
        else: # dx < 0
            return ('v', x, y) # West


if __name__ == '__main__':
    # for test
    re = pvl_to_loc(np.array([3.02, 5.31]), 10, 10)
=======
'''
Noted on March 14, 2023

This file is to perform transformation between ID of bins and location.

Here we defined these 3 terms in order to definitely illustrate what these functions aim 
  at doing.

xbins represents the number of bins on dimension x, while ybins represents the number of 
  bins on dimension y.

1. Bin Index (termed as idx): 
    The absolute Index of certain bin. If we define the value space into (xbin, ybin), 
    that the bin ID must belong to {1, 2, ..., xbin*ybin}, something like:

      M+1    M+2    ...   xbin*ybin 
       .      .     ...       .
       .      .     ...       .
       .      .     ...       .
    xbin+1  xbin+2  ...    xbin*2
       1      2     ...     xbin

2. Bin coordinate (abbreviated as loc):
    The x, y coordinate of where a certain bin locates at.

  (0, ybin-1) (1, ybin-1) ... (xbin-1, ybin-1)
       .           .      ...        .
       .           .      ...        .
       .           .      ...        .   
    (0, 1)      (1, 1)    ...   (xbin-1, 1)
    (0, 0)      (1, 0)    ...   (xbin-1, 0)

3. Precise value coordinate (termed as PVL/pvl)
    The precise recorded data (usually 2 dimension)

    For example, if the data is 2D spatial coordinate, it might be (10.843 cm, 85.492 cm)

    If the data is a kind of conjunctively paired data, such as data that jointly combined 
      voice frequencies and 1D spatial coordinate together, it might be (43.154 cm, 5185 Hz)

    It represents precise value of recorded data.


Function developed here are generally for transformation between these 3 forms of data.

Note that the transformation between loc and idx are reversible, but the transformation 
  from pvl to loc/idx are irreversible (for this kind of transformaiton will lose pretty 
  much precise information about the value location)

For this reason, only 4 function: 
                idx_to_loc, loc_to_idx, pvl_to_loc, pvl_to_idx 
  are available.
'''

import numpy as np
import warnings

from mazepy.behav.grid import GridSize, GridBasic


"""
Check whether a bin locates at the border of a maze.
"""
def isNorthBorder(BinID, xbin = 12, ybin = 12):
    if (BinID-1) // xbin == ybin - 1:
        return True
    else:
        return False

def isEastBorder(BinID, xbin = 12):
    if BinID % xbin == 0:
        return True
    else:
        return False

def isWestBorder(BinID, xbin = 12):
    if BinID % xbin == 1:
        return True
    else:
        return False

def isSouthBorder(BinID, xbin = 12):
    if BinID <= xbin:
        return True
    else:
        return False


"""
Define BinID class, BinCoordinate class, RawCoordinate class, NormCoordinate class
"""

class BinIDArray(GridBasic):
    def __init__(self, 
                 input_val: int or list or tuple or np.ndarray,
                 xbin: int, 
                 ybin: int
                ) -> None:
        super().__init__(xbin=xbin, ybin=ybin)
        self._values = None

        try:
            l = len(input)
            if type(input_val) is list or type(input_val) is tuple:
                self.values = np.array(input_val, dtype = np.int64)
            elif type(input_val) is np.ndarray:
                self.values = input_val.astype(np.int64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
        except:
            self.values = input_val

    @property
    def values(self):
        return self._value
    
    @values.setter
    def values(self, val):
        if type(val) is int:
            if val > self.xbin*self.ybin or val < 1:
                raise ValueError(f"{val} contains invalid value(s). Valid values belong to list [1, ..., {self.xbin*self.ybin}]")
            else:
                self._value = val
        else:
            if len(np.where((val > self.xbin*self.ybin)|(val < 1))[0]) != 0:
                raise ValueError(f"{val} contains invalid value(s). Valid values belong to list [1, ..., {self.xbin*self.ybin}]")
            else:
                self._value = val

    def __repr__(self):
        return 'BinIDArray ({}) at GridBasic ({} {})'.format(self._value, self.xbin, self.ybin)
        
    def __eq__(self, others) -> bool:
        return self._value == others.value
    
    def __ne__(self, others) -> bool:
        return self._value != others.value
    
    def __lt__(self, other) -> bool:
        return self._value < other.value
    
    def __gt__(self, other) -> bool:
        return self._value > other.value
    
    def to_binxy(self):
        return (self.values - 1) % self.xbin, (self.values - 1) // self.xbin

class BinCoordinateArray(GridBasic):
    def __init__(self, 
                 x: float or np.ndarray,
                 y: float or np.ndarray,
                 xbin: int, 
                 ybin: int
                ) -> None:
        super().__init__(xbin, ybin)
        self._x, self._y = None, None
        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, input_val: int or float or list or tuple or np.ndarray):
        try:
            l = len(input_val)
            if type(input_val) is list or type(input_val) is tuple:
                x = np.array(input_val, dtype = np.int64)
            elif type(input_val) is np.ndarray:
                x = input_val.astype(np.int64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((x > self.xbin-1)|(x < 0))[0]) != 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xbin-1}]")
                return
            else:
                self._x = x
        except:
            x = input_val
            if x > self.xbin-1 or x < 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xbin-1}]")
                return
            else:
                self._x = x
                
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, input_val):
        try:
            l = len(input_val)
            if type(input_val) is list or type(input_val) is tuple:
                y = np.array(input_val, dtype = np.int64)
            elif type(input_val) is np.ndarray:
                y = input_val.astype(np.int64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((y > self.ybin-1)|(y < 0))[0]) != 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ybin-1}]")
                return
            else:
                self._y = y
        except:
            y = input_val
            if y > self.ybin-1 or y < 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ybin-1}]")
                return
            else:
                self._y = y
        
    def to_binid(self):
        return self.x + self.y*self.ybin + 1
        

class RawCoordinateArray(GridSize):
    def __init__(self, 
                 x: float or np.ndarray,
                 y: float or np.ndarray,
                 xbin: int, 
                 ybin: int, 
                 xmax: float, 
                 ymax: float, 
                 xmin: int or float = 0, 
                 ymin: int or float = 0
                ) -> None:
        super().__init__(xbin, ybin, xmax, ymax, xmin, ymin)
        self._x, self._y = None, None
        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, input_val: int or float or tuple or np.ndarray):
        try:
            l = len(input_val)
            if type(input_val) is list or type(input_val) is tuple:
                x = np.array(input_val, dtype = np.float64)
            elif type(input_val) is np.ndarray:
                x = input_val.astype(np.float64)
            else:
                warnings.warn(f"Only int, float, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((x >= self.xmax)|(x < 0))[0]) != 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xmax})")
                return
            else:
                self._x = x

        except:
            x = input_val
            if x >= self.xmax or x < 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xmax})")
                return
            else:
                self._x = x          
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, input_val):
        try:
            l = len(input_val)
            if type(input_val) is list or type(input_val) is tuple:
                y = np.array(input_val, dtype = np.float64)
            elif type(input_val) is np.ndarray:
                y = input_val.astype(np.float64)
            else:
                warnings.warn(f"Only int, float, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((y >= self.ymax)|(y < 0))[0]) != 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ymax})")
                return
            else:
                self._y = y

        except:
            y = input_val
            if y >= self.ymax or y < 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ymax})")
                return
            else:
                self._y = y
    
    @property
    def Norm(self):
        return NormCoordinateArray((self.x - self.xmin) / (self.xmax + 0.0001 - self.xmin) * self.xbin, 
                                   (self.y - self.ymin) / (self.ymax + 0.0001 - self.ymin) * self.ybin, 
                                   xbin = self.xbin, ybin = self.ybin)
    
    def to_binid(self):
        return self.Norm.to_binid()
    
    def to_binxy(self):
        return self.Norm.to_binxy()
    

class NormCoordinateArray(GridBasic):
    def __init__(self, 
                 x: float or np.ndarray,
                 y: float or np.ndarray,
                 xbin: int, 
                 ybin: int
                ) -> None:
        super().__init__(xbin, ybin)
        self._x, self._y = None, None
        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, input_val: int or float or tuple or np.ndarray):
        try:
            l = len(input)
            if type(input_val) is list or type(input_val) is tuple:
                x = np.array(input_val, dtype = np.float64)
            elif type(input_val) is np.ndarray:
                x = input_val.astype(np.float64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((x >= self.xbin)|(x < 0))[0]) != 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xbin})")
                return
            else:
                self._x = x

        except:
            x = input_val
            if x >= self.xbin or x < 0:
                warnings.warn(f"{x} contains invalid value(s). Valid values belong to list [0, ..., {self.xbin})")
                return
            else:
                self._x = x
                return


    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, input_val):
        try:
            l = len(input)
            if type(input_val) is list or type(input_val) is tuple:
                y = np.array(input_val, dtype = np.float64)
            elif type(input_val) is np.ndarray:
                y = input_val.astype(np.float64)
            else:
                warnings.warn(f"Only int, list, tuple and np.ndarray object instead of {type(input_val)} are supported.")
                return

            if len(np.where((y >= self.ybin)|(y < 0))[0]) != 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ybin})")
                return
            else:
                self._y = y

        except:
            y = input_val
            if y >= self.ybin or y < 0:
                warnings.warn(f"{y} contains invalid value(s). Valid values belong to list [0, ..., {self.ybin})")
                return
            else:
                self._y = y
                return
        
    def to_binid(self) -> np.ndarray:
        if type(self.x) is int or type(self.x) is float:
            return int(self.x) + int(self.y)*self.xbin + 1
        else:
            return ((self.x // 1) + (self.y // 1)*self.xbin + 1).astype(np.int64)
    
    def to_binxy(self) -> np.ndarray:
        if type(self.x) is int or type(self.x) is float:
            return int(self.x), int(self.y)
        else:
            return (self.x // 1).astype(np.int64), (self.y // 1).astype(np.int64)
        




# =============================================================================================== 
# =============================================================================================== 
# =============================================================================================== 
# =============================================================================================== 




def pvl_to_idx(x: int or float or tuple or np.ndarray, 
               y: int or float or tuple or np.ndarray, 
               xbin: int, 
               ybin: int,
               xmax: float or int, 
               ymax: float or int, 
               xmin: float or int = 0,
               ymin: float or int = 0
              ) -> np.ndarray:
    """
    Parameter
    ---------
    x, y: int or float or tuple or numpy.ndarray object, required.
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xmin: float, optional, default: 0
    ymin: float, optional, default: 0

    Return
    ------
    idx: int or numpy.ndarray 1d vector, containing index/indices of a collection of bin(s)

    Note
    ----
    Transform precise value location into bin index.
    """
    Arr = RawCoordinateArray(x, y, xbin=xbin, ybin=ybin, xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin)
    return Arr.to_binid()


# transform 960cm position data into nx*nx form
def pvl_to_loc(x: int or float or tuple or np.ndarray, 
               y: int or float or tuple or np.ndarray, 
               xbin: int,
               ybin: int,
               xmax: float or int, 
               ymax: float or int, 
               xmin: float or int = 0,
               ymin: float or int = 0
              ) -> np.ndarray:
    """
    Parameter
    ---------
    x, y: int or float or tuple or numpy.ndarray object, required.
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xmin: float, optional, default: 0
    ymin: float, optional, default: 0

    Return
    ------
    coordinate: int or numpy.ndarray object with a shape of (2, T), containing bin location(s) of a collection of bin(s)

    Note
    ----
    Transform precise value location into bin location.
    """
    Arr = RawCoordinateArray(x, y, xbin=xbin, ybin=ybin, xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin)
    return Arr.to_binxy()

def loc_to_idx(x: np.ndarray or int, y: np.ndarray or int, xbin: int, ybin: int) -> np.ndarray:
    '''
    Parameter
    ---------
    x, y: The bin coordinate.
    xbin, ybin: int, required
        The total bin number of dimension xcoordinate

    Return
    ------
    idx: int or numpy.ndarray 1d vector, the bin index

    Note
    ----
    Transform bin location into bin index.
    '''
    Arr = BinCoordinateArray(x, y, xbin=xbin, ybin=ybin)
    return Arr.to_binid()


def idx_to_loc(idx: int or np.ndarray, xbin: int, ybin: int) -> np.ndarray:
    '''
    Parameter
    ---------
    idx: int or np.ndarray 1d vector, required
        the bin index
    xbin, ybin: int, optional
        The total bin number of dimension x
        default: 10

    Return
    ------
    x, y: int or np.ndarray 1d vector, respectively. 
        that is the bin coordinate (x, y)

    Note
    ----
    Transform bin index into bin location. 
    '''
    Arr = BinIDArray(idx, xbin=xbin, ybin=ybin)
    return Arr.to_binxy()


def loc_to_edge(x: float, y: float, xbin: int, ybin: int) -> tuple or None:
    if (type(x) is float or type(x) is int) and (type(y) is float or type(y) is int):
        if x == int(x) and y != int(y):
            return ('v', int(x), int(y))
        elif x != int(x) and y == int(y):
            return ('h', int(x), int(y))
        else:
            return
    else:
        return


def nearest_edge(x: int or float or tuple or np.ndarray, 
               y: int or float or tuple or np.ndarray, 
               xbin: int,
               ybin: int,
               xmax: float or int, 
               ymax: float or int, 
               xmin: float or int = 0,
               ymin: float or int = 0
              ) -> tuple:
    '''
    Parameter
    ---------
    x, y: int or float or tuple or numpy.ndarray object, required.
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xmin: float, optional, default: 0
    ymin: float, optional, default: 0

    Return
    ------
    tuple: (direction, x, y) of the edge

    Note
    ----
    Find the nearest edge to a precise value point.
    '''
    Arr = RawCoordinateArray(x, y, xbin=xbin, ybin=ybin, xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin)
    xp, yp = Arr.Norm.x, Arr.Norm.y
    x, y = int(xp), int(yp)

    dy = yp - y - 0.5
    dx = xp - x - 0.5
    
    if np.abs(dx) <= np.abs(dy):
        if dy >= 0:
            return ('h', x, y+1)  # North
        else: # dy < 0
            return ('h', x, y)  # South
    else: # np.abs(dx) > np.abs(dy):
        if dx >= 0:  
            return ('v', x+1, y) # East
        else: # dx < 0
            return ('v', x, y) # West


if __name__ == '__main__':
    # for test
    re = pvl_to_loc(np.array([3.02, 5.31]), 10, 10)
>>>>>>> bebfe805e63d3226d6196e1b8ff4c78089a1de31
    print(re, re.dtype)