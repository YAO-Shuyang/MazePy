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

2. Bin location (abbreviated as loc):
    The x, y location of where a certain bin locates at.

  (0, ybin-1) (1, ybin-1) ... (xbin-1, ybin-1)
       .           .      ...        .
       .           .      ...        .
       .           .      ...        .   
    (0, 1)      (1, 1)    ...   (xbin-1, 1)
    (0, 0)      (1, 0)    ...   (xbin-1, 0)

3. Precise value location (termed as PVL/pvl)
    The precise recorded data (usually 2 dimension)

    For example, if the data is 2D spatial location, it might be (10.843 cm, 85.492 cm)

    If the data is a kind of conjunctively paired data, such as data that jointly combined 
      voice frequencies and 1D spatial location together, it might be (43.154 cm, 5185 Hz)

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


def pvl_to_idx(prec_value_loc: np.ndarray, xmax: float or int, ymax: float or int, xbin: int = 10, ybin: int = 10):
    '''
    Parameter
    ---------
    prec_value_loc: numpy.ndarray object, required.
        with size of (2, T). If the first dimension is not 2, it will raise an error.
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10

    Return
    ------
    idx: int or numpy.ndarray 1d vector, containing index/indices of a collection of bin(s)

    Note
    ----
    Transform precise value location into bin index.
    '''
    Res = pvl_to_loc(prec_value_loc = prec_value_loc, xmax = xmax, ymax = ymax, xbin = xbin, ybin = ybin)
    return loc_to_idx(cell_x = Res[0, :], cell_y = Res[1, :], xbin = xbin)

# transform 960cm position data into nx*nx form
def pvl_to_loc(prec_value_loc: np.ndarray, xmax: float or int, ymax: float or int, xbin: int = 10, ybin: int = 10):
    '''
    Parameter
    ---------
    prec_value_loc: numpy.ndarray object, required.
        with size of (2, T). If the first dimension is not 2, it will raise an error.
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10

    Return
    ------
    loc: int or numpy.ndarray object with a shape of (2, T), containing bin location(s) of a collection of bin(s)

    Note
    ----
    Transform precise value location into bin location.
    '''
    assert prec_value_loc.shape[0] == 2
    try:
        prec_value_loc.shape[1]
        loc = np.zeros_like(prec_value_loc)
        loc[0, :] = prec_value_loc[0, :] / xmax * xbin // 1
        loc[1, :] = prec_value_loc[1, :] / ymax * ybin // 1
        loc = loc.astype(dtype = np.int64)
        return loc
    except:
        return np.array([int(prec_value_loc[0] / xmax * xbin), int(prec_value_loc[1] / ymax * ybin)])

def loc_to_idx(cell_x: np.ndarray or int, cell_y: np.ndarray or int, xbin: int = 10):
    '''
    Parameter
    ---------
    cell_x: int or np.ndarray 1d vector, required
        x index of bin location
    cell_y: int or np.ndarray 1d vector, required
        y index of bin location
    xbin: int, optional
        The total bin number of dimension x
        default: 10

    Return
    ------
    idx: int or numpy.ndarray 1d vector, the bin index

    Note
    ----
    Transform bin location into bin index.
    '''
    idx = cell_x + cell_y * xbin + 1
    return idx


def idx_to_loc(idx: np.ndarray or int, xbin: int = 10):
    '''
    Parameter
    ---------
    idx: int or np.ndarray 1d vector, required
        the bin index
    xbin: int, optional
        The total bin number of dimension x
        default: 10

    Return
    ------
    cell_x, cell_y: int or np.ndarray 1d vector, respectively. 
        Elements of the bin location -- (cell_x, cell_y)

    Note
    ----
    Transform bin index into bin location. 
    '''
    cell_x = (idx-1) % xbin
    cell_y = (idx-1) // xbin
    return cell_x, cell_y


def idx_to_edge(idx: np.ndarray or int, xbin: int):
    '''
    Parameter
    ---------
    idx: np.ndarray or int object, required
        The bin ID

    xbin: int, required
        The total bin number of dimension x
    
    Return
    ------
    A dict contains information of four edges that suround the bin.

    Note
    ----
    Find the four edge of a certain bin.
    '''
    x, y = idx_to_loc(idx, xbin = xbin)

    return {'north': np.array([x, y+1], dtype = np.int), 
            'south': np.array([x, y], dtype = np.int), 
            'east': np.array([x+1, y], dtype = np.int),
            'west': np.array([x, y], dtype = np.int)}


def pvl_to_edge(prec_value_loc: np.ndarray, xmax: float or int, ymax: float or int, xbin: int = 10, ybin: int = 10) -> tuple:
    '''
    Parameter
    ---------
    prec_value_loc: np.ndarray, required.
        with size of (2,).
    xmax: float, required
        the max value of xmax
    ymax: float, required
        the max value of ymax
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10

    Return
    ------
    tuple: (direction, x, y) of the edge

    Note
    ----
    Find the nearest edge to a precise value point.
    '''
    x, y = pvl_to_loc(prec_value_loc=prec_value_loc, xmax=xmax, ymax=ymax, xbin=xbin, ybin=ybin)
    xp, yp = prec_value_loc[0]/xmax*xbin, prec_value_loc[1]/ymax*ybin

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
    print(re, re.dtype)