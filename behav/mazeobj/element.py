"""
Provide 3 basic elements for maze design: Bin, Edge, Point

Their basic properties are listed here:

1. Bin: defined by (x, y)
        - four edges -> edge objects
        - four corners -> point objects

2. Point: defined by (x, y)
        - four edges -> edge objects
        - four bins -> bin objects

3. Edge: defined by (dir, x, y)   dir: direction, 'v' or 'h'
        - two ends -> point objects
        - two side bins -> bin objects
"""

import numpy as np

from mazepy.behav.gridbin import GridBasic
from mazepy.behav.transloc import loc_to_idx



class Elements(GridBasic):
    def __init__(self, xbin: int, ybin: int, x: int, y:int) -> None:
        """
        Parameter
        ---------
        xbin: int, required
            Total number of the bins at dimension x.
        ybin: int, required
            Total number of the bins at dimension y.
        x: int, required
            First value of the cordinate.
        y: int, required
            Second value of the cordinate. Togather with x, they make up the
              cordinate (x,y) and it is sufficient to point out a bin.        
        """
        super().__init__(xbin=xbin, ybin=ybin)

        if x < 0 or x > xbin:
            raise ValueError(f"""{x} is overflow! The value to initiate Element
                             object should at least belong to [0 {xbin+1})""")
        if y < 0 or y > ybin:
            raise ValueError(f"""{y} is overflow! The value to initiate Element
                             object should at least belong to [0 {ybin+1})""")
        self._x = x
        self._y = y
    
    @property
    def cordinate(self):
        return (self._x, self._y)
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y



class Bin(Elements):
    def __init__(self, xbin: int, ybin: int, x: int, y:int) -> None:
        """
        Parameter
        ---------
        xbin: int, required
            Total number of the bins at dimension x.
        ybin: int, required
            Total number of the bins at dimension y.
        x: int, required
            First value of the cordinate.
        y: int, required
            Second value of the cordinate. Togather with x, they make up the
              cordinate (x,y) and it is sufficient to point out a bin.    
        """
        super().__init__(xbin=xbin, ybin=ybin, x=x, y=y)
        if x == xbin:
            raise ValueError(f"""{x} is overflow! The value to initiate Bin
                             object should at least belong to [0 {xbin})""")
        if y == ybin:
            raise ValueError(f"""{y} is overflow! The value to initiate Bin
                             object should at least belong to [0 {ybin})""")        
        self._find_four_corners()
        self._find_four_edges()


    def _find_four_corners(self):
        x, y = self._x, self._y
        self._ul = (x,   y+1)
        self._ur = (x+1, y+1)
        self._bl = (x,   y)
        self._br = (x+1, y)

    def _find_four_edges(self):
        x, y = self._x, self._y
        self._N = ('h', x, y+1)
        self._S = ('h', x, y)
        self._E = ('v', x+1, y)
        self._W = ('v', x, y)

    @property
    def id(self):
        return loc_to_idx(cell_x = self._x, cell_y = self._y, xbin = self.xbin)

    @property
    def ul(self):
        return self._ul
    
    @property
    def ur(self):
        return self._ur
    
    @property
    def bl(self):
        return self._bl

    @property
    def br(self):
        return self._br

    @property
    def N(self):
        return self._N

    @property
    def S(self):
        return self._S

    @property
    def E(self):
        return self._E
    
    @property
    def W(self):
        return self._W



class Edge(Elements):
    def __init__(self, xbin: int, ybin: int, x: int, y:int, dirc: str) -> None:
        """
        Parameter
        ---------
        xbin: int, required
            The total bin number of dimension x
        ybin: int, required
            The total bin number of dimension y
        x: int, required
            The row that the edge lays on.
        y: int, required
            The column that the edge lays on.
        dirc: str, optional
            Values in {'v', 'h'} are available. 'v' represents vertical edges, while 'h' represents
              horizontal ones.
        """
        super().__init__(xbin=xbin, ybin=ybin, x=x, y=y)

        if dirc not in ['v', 'h']:
            raise ValueError(f"{dirc} is an invalid value. Only 'v' and 'h' are accepted.")
        
        if x == xbin and dirc == 'h':
            raise ValueError(f"""{x} is overflow! The value to initiate Bin
                             object should at least belong to [0 {xbin}), 
                             when the direction is {dirc}""")
        if y == ybin and dirc == 'v':
            raise ValueError(f"""{y} is overflow! The value to initiate Edge object should at least belong to [0 {ybin}), when the direction is {dirc}""")   
        
        self._dir = dirc
        self._find_two_ends()
        self._find_two_bins()

    @property
    def dir(self):
        return self._dir
    
    def _find_two_ends(self):
        x, y = self._x, self._y
        if self._dir == 'v':
            self._p1 = (x, y)
            self._p2 = (x, y+1)
        elif self._dir == 'h':
            self._p1 = (x, y)
            self._p2 = (x+1, y)

    def _find_two_bins(self):
        x, y = self._x, self._y
        if self._dir == 'v':
            self._bin1 = (x-1, y) if x >= 1 else None
            self._bin2 = (x, y) if x <= self.xbin-1 else None
        elif self._dir == 'h':
            self._bin1 = (x, y-1) if y >= 1 else None
            self._bin2 = (x, y) if y <= self.ybin-1 else None
    
    @property
    def p1(self):
        return self._p1
    
    @property
    def p2(self):
        return self._p2
    
    @property
    def bin1(self):
        return self._bin1
    
    @property
    def bin2(self):
        return self._bin2
    



class Point(Elements):
    def __init__(self, xbin: int, ybin: int, x: int, y: int) -> None:
        """
        Parameter
        ---------
        xbin: int, required
            The total bin number of dimension x
        ybin: int, required
            The total bin number of dimension y
        x: int, required
            The row that the edge lays on.
        y: int, required
            The column that the edge lays on.
        dirc: str, optional
            Values in {'v', 'h'} are available. 'v' represents vertical edges, while 'h' represents
              horizontal ones.
        """
        super().__init__(xbin, ybin, x, y)

        self._find_four_corners()
        self._find_four_edges()


    def _find_four_corners(self):
        x, y = self._x, self._y
        self._ul = (x,   y+1) if x >= 1           and y <= self.ybin-1 else None
        self._ur = (x+1, y+1) if x <= self.xbin-1 and y <= self.ybin-1 else None
        self._bl = (x,   y)   if x >= 1           and y >= 1           else None
        self._br = (x+1, y)   if x <= self.xbin-1 and y >= 1           else None

    def _find_four_edges(self):
        x, y = self._x, self._y
        self._N = ('v', x, y)   if y <= self.ybin-1 else None
        self._S = ('v', x, y-1) if y >= 1           else None
        self._E = ('h', x, y)   if x <= self.xbin-1 else None
        self._W = ('h', x-1, y) if x >= 1           else None

    @property
    def ul(self):
        return self._ul
    
    @property
    def ur(self):
        return self._ur
    
    @property
    def bl(self):
        return self._bl

    @property
    def br(self):
        return self._br

    @property
    def N(self):
        return self._N

    @property
    def S(self):
        return self._S

    @property
    def E(self):
        return self._E
    
    @property
    def W(self):
        return self._W