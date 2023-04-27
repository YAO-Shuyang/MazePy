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

from mazepy.behav.grid import GridBasic
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
        
        if x != int(x):
            raise ValueError(f"x should be an integer, but receive a float({x}) instead.")
        
        if y != int(y):
            raise ValueError(f"y should be an integer, but receive a float({y}) instead.")

        self._x = int(x)
        self._y = int(y)
    
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

    def place_wall(self, Graph: dict):
        """
        To determine whether the four edges of the bin will be placed with a wall.

        Parameter
        ---------
        Graph: dict, required.
            The graph of the maze contains the fundamental information about the 
              maze structure that is need for placing the walls.
        """
        surr = Graph[self.id]
        self._WWall, self._SWall, self._EWall, self._WWall = True, True, True, True

        for s in surr:
            if s == self.id + self.xbin:
                self._NWall = False
                continue
            if s == self.id + 1:
                self._EWall = False
                continue
            if s == self.id - 1:
                self._WWall = False
                continue
            if s == self.id - self.xbin:
                self._SWall = False
                continue

    @property
    def id(self):
        return loc_to_idx(self._x, self._y, xbin=self.xbin, ybin=self.ybin)

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
    
    @property
    def NWall(self):
        return self._NWall
    
    @property
    def SWall(self):
        return self._SWall
    
    @property
    def EWall(self):
        return self._EWall
    
    @property
    def WWall(self):
        return self._WWall
    


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
    
    def place_wall(self, Graph: dict) -> bool:
        """
        To determine whether this edge will be placed with a wall.

        Parameter
        ---------
        Graph: dict, required.
            The graph of the maze contains the fundamental information about the 
              maze structure that is need for placing the walls.
        """
        if self.bin1 is None or self.bin2 is None:
            self._Wall = True
            return True
        
        id1 = loc_to_idx(self.bin1[0], self.bin1[1], xbin=self.xbin, ybin=self.ybin)
        id2 = loc_to_idx(self.bin2[0], self.bin2[1], xbin=self.xbin, ybin=self.ybin)

        if id2 in Graph[id1]:
            self._Wall = False
            return False
        else:
            self._Wall = True
            return True

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
    
    @property
    def Wall(self):
        return self._Wall


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

    def place_wall(self, Graph: dict):
        """
        To determine whether the four connected edges of the point will be placed 
          with a wall.

        Parameter
        ---------
        Graph: dict, required.
            The graph of the maze contains the fundamental information about the 
              maze structure that is need for placing the walls.
        """
        NEdge = Edge(self.xbin, self.ybin, x = self.x, y = self.y, dirc = 'v') if self._N is not None else None
        SEdge = Edge(self.xbin, self.ybin, x = self.x, y = self.y-1, dirc = 'v') if self._S is not None else None
        EEdge = Edge(self.xbin, self.ybin, x = self.x, y = self.y, dirc = 'h') if self._E is not None else None
        WEdge = Edge(self.xbin, self.ybin, x = self.x-1, y = self.y, dirc = 'h') if self._W is not None else None
        
        self._NWall = NEdge.place_wall(Graph=Graph) if self._N is not None else None
        self._SWall = SEdge.place_wall(Graph=Graph) if self._S is not None else None
        self._EWall = EEdge.place_wall(Graph=Graph) if self._E is not None else None
        self._WWall = WEdge.place_wall(Graph=Graph) if self._W is not None else None


    def is_passable(self, Graph: dict, x: float, y:float):
        """ 
        To determine whether a beam emitted from point (x,y) (named as emit point) 
          would pass the point where connected with wall(s) (named as corner point).
        If only the north and east edge that connected to the corner point would be 
          placed with a wall, these four contitions should be discussed respectively:
          (Note that walls are denoted as '|' and '——', whereas the edge that would 
          not be placed with a wall is represented by a arrow pointing to different 
          direction. (x,y) refers to emit point):

        1. pass from southeast:
                     |
                 ←-self——
                     ↓
                        (x,y)    ->  can pass the point
        2. pass from southwest:
                     |
                 ←-self——
                     ↓
             (x,y)               ->  cannot pass the point
        3. pass from Northwest:
             (x,y)
                     |
                 ←-self——
                     ↓           ->  can pass the point
        4. pass from Northeast:
                        (x,y)
                     |
                 ←-self——
                     ↓           ->  cannot pass the point
        """
        if self.is_corner_point(Graph=Graph) == False:
            return False
        
        assert x >= 0 and x < self.xbin and y >= 0 and y < self.ybin
        
        # emit point is overlapped with corner point
        if x == self.x and y == self.y:
            return True
        #                                          __
        # Walls are placed at a 90° angle (|__ or    |), and every emit point at the second quadrant or the
        # forth quadrant can pass the corner point, whereas every emit point at the first and third quadrant
        # can not pass the corner point.
        if self.NWall and self.EWall:
            if (self.x - x)*(self.y - y) < 0:
                return True
            elif (self.x - x)*(self.y - y) > 0:
                return False
            elif self.y == y:
                if self.x > x:
                    return False
                else: #self.x < x
                    return True
            else: # self.x == x:
                if self.y > y:
                    return False
                else:
                    return True
                
        if self.SWall and self.WWall:
            if (self.x - x)*(self.y - y) < 0:
                return True
            elif (self.x - x)*(self.y - y) > 0:
                return False
            elif self.y == y:
                if self.x > x:
                    return True
                else: #self.x < x
                    return False
            else: # self.x == x:
                if self.y > y:
                    return True
                else:
                    return False            
            
        #                                          __   
        # Walls are placed at a 90° angle (__| or |   ), and every emit point at the first and third quadrant
        # can pass the corner point, whereas every emit point at the second and forth quadrant cannot pass the 
        # corner point.
        if self.NWall and self.WWall:
            if (self.x - x)*(self.y - y) > 0:
                return True
            elif (self.x - x)*(self.y - y) < 0:
                return False
            elif self.y == y:
                if self.x > x:
                    return True
                else: #self.x < x
                    return False
            else: # self.x == x:
                if self.y > y:
                    return False
                else: return True # self.y < y

        if self.SWall and self.EWall:
            if (self.x - x)*(self.y - y) > 0:
                return True
            elif (self.x - x)*(self.y - y) < 0:
                return False
            elif self.y == y:
                if self.x > x:
                    return False
                else: #self.x < x
                    return True
            else: # self.x == x:
                if self.y > y:
                    return True
                else: return False # self.y < y

        # The rest conditions are those points that have got only 1 connected wall or even haven't got a connected
        # wall. Beams from emit point can easily pass the corner point.
        B1 = self.x == x and self.y < y and self.SWall == True
        B2 = self.x == x and self.y > y and self.NWall == True
        B3 = self.y == y and self.x < x and self.WWall == True
        B4 = self.y == y and self.x > x and self.EWall == True

        if B1 or B2 or B3 or B4:
            return False
        else:
            return True
    
    def is_corner_point(self, Graph: dict) -> bool:
        """
        Return
        ------
        bool, whether the point is a corner point.
        """
        self.place_wall(Graph=Graph)

        # Points at the border of the environment will never be passed.
        if self.NWall is None or self.SWall is None or self.EWall is None or self.WWall is None:
            return False

        if (self.EWall and self.WWall) or (self.NWall and self.SWall):
            return False
        
        if (self.EWall and self.NWall) or (self.WWall and self.SWall) or (self.NWall and self.WWall) or (self.SWall and self.EWall):
            return True
        
        return True
    
    @property
    def wall_num(self):
        self._wall_num = 0
        for w in [self._NWall, self._SWall, self._EWall, self._WWall]:
            if w:
                self._wall_num += 1
        return self._wall_num

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
    
    @property
    def NWall(self):
        return self._NWall
    
    @property
    def SWall(self):
        return self._SWall
    
    @property
    def EWall(self):
        return self._EWall
    
    @property
    def WWall(self):
        return self._WWall
    
    @property
    def is_corner(self):
        return self._is_corner