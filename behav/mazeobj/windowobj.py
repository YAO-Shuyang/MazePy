<<<<<<< HEAD
'''
This file provides edge class for the GUI. 

Edges should have several properties:
  0. A edge bridges two bins.
  1. They could be selected as the wall if a mouse event occurs around it.
  2. If they are selected, they should change their color to reflect these change.
  3. The selections are reversible, that is, they could be canceled by reclicking on it.
  4. Once a wall is selected, the connection between 2 bins is blocked.

'''

# importing pyglet module
import pyglet
from pyglet import shapes

import time
import numpy as np
import copy as cp

from mazepy.behav.grid import GridBasic
from mazepy.behav.transloc import loc_to_idx
from mazepy.behav.element import Bin, Point, Edge

UNSELECTED_COLOR = (204, 255, 204)
SELECTED_COLOR = (0, 0, 0)


class WindowsCalculator(GridBasic):
    def __init__(self, 
                 xbin: int,
                 ybin: int,
                 ur_val: tuple,
                 ul_val: tuple,
                 br_val: tuple,
                 bl_val: tuple
                ) -> None:
        """
        Parameter
        ---------
        ur, ul, br, bl: tuple, 
            The precise (x, y) coordinate of the four corners on the screen of the GUI.
            ur: upper right     ul: upper left      br: bottom right    bf: bottom left

        Note
        ----
        Bridge the coordinate of Bin, Point, Edge and the precise coordinate on the GUI screen.
        Provide functions to realize this transformation.
        """
        super().__init__(xbin=xbin, ybin=ybin)
        self._ul = ul_val
        self._ur = ur_val
        self._bl = bl_val
        self._br = br_val
        self._xrange = (self._br[0] - self._bl[0])
        self._yrange = (self._ur[1] - self._br[1])
        self._xlen = self._xrange / self.xbin # length per bin at dimension x
        self._ylen = self._yrange / self.ybin # length per bin at dimension y

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
    def bin_size(self):
        return (self._xlen, self._ylen)
    
    @property
    def xrange(self):
        return self._xrange
    
    @property
    def yrange(self):
        return self._yrange
    
    @property
    def xlen(self):
        return self._xlen
    
    @property
    def ylen(self):
        return self._ylen
    
    def cor_to_scr(self, x: int, y:int) -> tuple:
        """
        Parameter
        ---------
        x, y: int, required
            The coordinate that need to transform.
        """
        return x*self.xlen + self.bl[0], y*self.ylen + self.bl[1]


class WindowsWall(Edge):
    '''
    Note
    ----
    Define the properties of internal structure
    '''
    def __init__(self, 
                 xbin: int,
                 ybin: int,
                 x: int,
                 y: int,
                 dirc: str,
                 calculator: WindowsCalculator = None,
                 **kwargs
                ) -> None:
        '''
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
        calculator: WindowsCalculator calss:
            For the convience of coordinate transformation.
        dirc: str, optional
            Values in {'v', 'h'} are available. 'v' represents vertical edges, while 'h' represents
              horizontal ones.
        '''
        super().__init__(xbin=xbin, ybin=ybin, x=x, y=y, dirc = dirc)
        self._state = 0 # the state of the edge: 0 -> no wall; 1 -> has a wall.
        self._calculator = calculator
        self._color = UNSELECTED_COLOR
        self._time = time.time()
        self._two_bins_id_on_gui()
        self._two_ends_coordinate_on_gui()

    def _two_bins_id_on_gui(self):
        """(x,y) coordinate to bin id.
        """
        self._id1 = loc_to_idx(self._bin1[0], self._bin1[1], self.xbin, self.ybin) if self._bin1 is not None else None
        self._id2 = loc_to_idx(self._bin2[0], self._bin2[1], self.xbin, self.ybin) if self._bin2 is not None else None

    @property
    def id1(self):
        return self._id1
    
    @property
    def id2(self):
        return self._id2

    def _two_ends_coordinate_on_gui(self):
        """
        Calculate the precise (x,y) coordinate of the two ends of the edge on the screen of 
          the GUI.
        """
        self._x1, self._y1 = self._calculator.cor_to_scr(self.p1[0], self.p1[1])
        self._x2, self._y2 = self._calculator.cor_to_scr(self.p2[0], self.p2[1])
    
    @property
    def p_aft1(self):
        """Point1 after transformation
        """
        return (self._x1, self._y1)
    
    @property
    def p_aft2(self):
        """Point2 after transformation
        """
        return (self._x2, self._y2)    

    def plot_line_on_batch(self, 
                           batch: pyglet.graphics.Batch,
                           **kwargs
                          ) -> shapes.Line:
        '''
        Parameter
        ---------
        batch: pyglet.graphics.Batch object, required
            Contains all of the edges.

        Note
        ----
        To plot the line on the GUI.
        '''
        line = shapes.Line(self._x1, self._y1, self._x2, self._y2, color = self._color, batch = batch, **kwargs)
        line.opacity = 255
        self.line = line
        return line

    def state_change(self, 
                     batch: pyglet.graphics.Batch,
                     Graph: dict,
                     **kwargs
                    ) -> dict:
        '''
        Parameter
        ---------
        batch: pyglet.graphics.Batch object, required
            Contains all of the edges.
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.
            
        Return
        ------
        A pyglet.shapes.Line object
        '''
        if time.time() - self._time >= 0.5:
            self._state = 1 - self._state
            self._color = UNSELECTED_COLOR if self._state == 0 else SELECTED_COLOR
            self._time = time.time()
            Graph = self._modify_graph(Graph=Graph)
            line = self.plot_line_on_batch(batch = batch, **kwargs)
            return Graph
        else:
            return Graph
        
    def _modify_graph(self,
                     Graph: dict
                    ) -> dict:
        '''
        Parameter
        ---------
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.

        Return
        ------
        A graph dict.
        '''
        if self.id1 is None or self.id2 is None:
            return Graph
        else:
            if self._state == 1:
                return self._break_connection(Graph, self.id1, self.id2)
            elif self._state == 0:
                return self._reconnection(Graph, self.id1, self.id2)
            else:
                raise ValueError(f"{self._state} is not a valid state value.")
        
    def _break_connection(self,
                          Graph: dict,
                          bin1: int,
                          bin2: int
                         ) -> dict:
        '''
        Parameter
        ---------
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.
        bin1, bin2: int, required
            bin1 and bin2 are the ID of the two bins that at the two sides of the line.
        '''
        d1 = np.where(Graph[bin1] == bin2)[0]
        d2 = np.where(Graph[bin2] == bin1)[0]

        Graph[bin1] = np.delete(Graph[bin1], d1)
        Graph[bin2] = np.delete(Graph[bin2], d2)
        return Graph
    
    def _reconnection(self,
                      Graph: dict,
                      bin1: int,
                      bin2: int
                     ) -> dict:
        '''
        Parameter
        ---------
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.
        bin1: int, required
        bin2: int, required
            bin1 and bin2 are the two bin that at the two sides of the line.
        '''
        Graph = cp.deepcopy(Graph)
        Graph[bin1] = np.append(Graph[bin1], bin2)
        Graph[bin2] = np.append(Graph[bin2], bin1)
        return Graph


class WindowsBin(Bin):
    def __init__(self, 
                 xbin: int, 
                 ybin: int, 
                 x: int, 
                 y: int, 
                 calculator: WindowsCalculator = None
                ) -> None:
        super().__init__(xbin, ybin, x, y)

        self._state = 1
        # 0 -> (This bin is) selected to be abandoned; 
        # 1 -> (This bin is) kept as a part of the environment.
        self._calculator = calculator
        self._time = time.time()
    
    def state_change(self, 
                     batch: pyglet.graphics.Batch,
                     occu_map: np.ndarray,
                     **kwargs
                    ) -> np.ndarray:
        if time.time() - self._time >= 0.5:
            self._state = 1 - self._state
            if self._state == 0:
                self._plot_diagonal(batch=batch, **kwargs)
                occu_map[self.id-1] = np.nan
            elif self._state == 1:
                self._erase_diagonal()
                occu_map[self.id-1] = 0
            self._time = time.time()
            return occu_map
        else:
            return occu_map
        
    def _plot_diagonal(self, batch: pyglet.graphics.Batch, **kwargs):
        x1, y1 = self._calculator.cor_to_scr(self.bl[0], self.bl[1])
        x2, y2 = self._calculator.cor_to_scr(self.ur[0], self.ur[1])
        self.Line1 = shapes.Line(x1, y1, x2, y2, batch = batch, color = SELECTED_COLOR, **kwargs)

        x1, y1 = self._calculator.cor_to_scr(self.br[0], self.br[1])
        x2, y2 = self._calculator.cor_to_scr(self.ul[0], self.ul[1])
        self.Line2 = shapes.Line(x1, y1, x2, y2, batch = batch, color = SELECTED_COLOR, **kwargs)

    def _erase_diagonal(self):
        self.Line1.delete()
=======
'''
This file provides edge class for the GUI. 

Edges should have several properties:
  0. A edge bridges two bins.
  1. They could be selected as the wall if a mouse event occurs around it.
  2. If they are selected, they should change their color to reflect these change.
  3. The selections are reversible, that is, they could be canceled by reclicking on it.
  4. Once a wall is selected, the connection between 2 bins is blocked.

'''

# importing pyglet module
import pyglet
from pyglet import shapes

import time
import numpy as np
import copy as cp

from mazepy.behav.grid import GridBasic
from mazepy.behav.transloc import loc_to_idx
from mazepy.behav.element import Bin, Point, Edge

UNSELECTED_COLOR = (204, 255, 204)
SELECTED_COLOR = (0, 0, 0)


class WindowsCalculator(GridBasic):
    def __init__(self, 
                 xbin: int,
                 ybin: int,
                 ur_val: tuple,
                 ul_val: tuple,
                 br_val: tuple,
                 bl_val: tuple
                ) -> None:
        """
        Parameter
        ---------
        ur, ul, br, bl: tuple, 
            The precise (x, y) coordinate of the four corners on the screen of the GUI.
            ur: upper right     ul: upper left      br: bottom right    bf: bottom left

        Note
        ----
        Bridge the coordinate of Bin, Point, Edge and the precise coordinate on the GUI screen.
        Provide functions to realize this transformation.
        """
        super().__init__(xbin=xbin, ybin=ybin)
        self._ul = ul_val
        self._ur = ur_val
        self._bl = bl_val
        self._br = br_val
        self._xrange = (self._br[0] - self._bl[0])
        self._yrange = (self._ur[1] - self._br[1])
        self._xlen = self._xrange / self.xbin # length per bin at dimension x
        self._ylen = self._yrange / self.ybin # length per bin at dimension y

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
    def bin_size(self):
        return (self._xlen, self._ylen)
    
    @property
    def xrange(self):
        return self._xrange
    
    @property
    def yrange(self):
        return self._yrange
    
    @property
    def xlen(self):
        return self._xlen
    
    @property
    def ylen(self):
        return self._ylen
    
    def cor_to_scr(self, x: int, y:int) -> tuple:
        """
        Parameter
        ---------
        x, y: int, required
            The coordinate that need to transform.
        """
        return x*self.xlen + self.bl[0], y*self.ylen + self.bl[1]


class WindowsWall(Edge):
    '''
    Note
    ----
    Define the properties of internal structure
    '''
    def __init__(self, 
                 xbin: int,
                 ybin: int,
                 x: int,
                 y: int,
                 dirc: str,
                 calculator: WindowsCalculator = None,
                 **kwargs
                ) -> None:
        '''
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
        calculator: WindowsCalculator calss:
            For the convience of coordinate transformation.
        dirc: str, optional
            Values in {'v', 'h'} are available. 'v' represents vertical edges, while 'h' represents
              horizontal ones.
        '''
        super().__init__(xbin=xbin, ybin=ybin, x=x, y=y, dirc = dirc)
        self._state = 0 # the state of the edge: 0 -> no wall; 1 -> has a wall.
        self._calculator = calculator
        self._color = UNSELECTED_COLOR
        self._time = time.time()
        self._two_bins_id_on_gui()
        self._two_ends_coordinate_on_gui()

    def _two_bins_id_on_gui(self):
        """(x,y) coordinate to bin id.
        """
        self._id1 = loc_to_idx(self._bin1[0], self._bin1[1], self.xbin, self.ybin) if self._bin1 is not None else None
        self._id2 = loc_to_idx(self._bin2[0], self._bin2[1], self.xbin, self.ybin) if self._bin2 is not None else None

    @property
    def id1(self):
        return self._id1
    
    @property
    def id2(self):
        return self._id2

    def _two_ends_coordinate_on_gui(self):
        """
        Calculate the precise (x,y) coordinate of the two ends of the edge on the screen of 
          the GUI.
        """
        self._x1, self._y1 = self._calculator.cor_to_scr(self.p1[0], self.p1[1])
        self._x2, self._y2 = self._calculator.cor_to_scr(self.p2[0], self.p2[1])
    
    @property
    def p_aft1(self):
        """Point1 after transformation
        """
        return (self._x1, self._y1)
    
    @property
    def p_aft2(self):
        """Point2 after transformation
        """
        return (self._x2, self._y2)    

    def plot_line_on_batch(self, 
                           batch: pyglet.graphics.Batch,
                           **kwargs
                          ) -> shapes.Line:
        '''
        Parameter
        ---------
        batch: pyglet.graphics.Batch object, required
            Contains all of the edges.

        Note
        ----
        To plot the line on the GUI.
        '''
        line = shapes.Line(self._x1, self._y1, self._x2, self._y2, color = self._color, batch = batch, **kwargs)
        line.opacity = 255
        self.line = line
        return line

    def state_change(self, 
                     batch: pyglet.graphics.Batch,
                     Graph: dict,
                     **kwargs
                    ) -> dict:
        '''
        Parameter
        ---------
        batch: pyglet.graphics.Batch object, required
            Contains all of the edges.
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.
            
        Return
        ------
        A pyglet.shapes.Line object
        '''
        if time.time() - self._time >= 0.5:
            self._state = 1 - self._state
            self._color = UNSELECTED_COLOR if self._state == 0 else SELECTED_COLOR
            self._time = time.time()
            Graph = self._modify_graph(Graph=Graph)
            line = self.plot_line_on_batch(batch = batch, **kwargs)
            return Graph
        else:
            return Graph
        
    def _modify_graph(self,
                     Graph: dict
                    ) -> dict:
        '''
        Parameter
        ---------
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.

        Return
        ------
        A graph dict.
        '''
        if self.id1 is None or self.id2 is None:
            return Graph
        else:
            if self._state == 1:
                return self._break_connection(Graph, self.id1, self.id2)
            elif self._state == 0:
                return self._reconnection(Graph, self.id1, self.id2)
            else:
                raise ValueError(f"{self._state} is not a valid state value.")
        
    def _break_connection(self,
                          Graph: dict,
                          bin1: int,
                          bin2: int
                         ) -> dict:
        '''
        Parameter
        ---------
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.
        bin1, bin2: int, required
            bin1 and bin2 are the ID of the two bins that at the two sides of the line.
        '''
        d1 = np.where(Graph[bin1] == bin2)[0]
        d2 = np.where(Graph[bin2] == bin1)[0]

        Graph[bin1] = np.delete(Graph[bin1], d1)
        Graph[bin2] = np.delete(Graph[bin2], d2)
        return Graph
    
    def _reconnection(self,
                      Graph: dict,
                      bin1: int,
                      bin2: int
                     ) -> dict:
        '''
        Parameter
        ---------
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.
        bin1: int, required
        bin2: int, required
            bin1 and bin2 are the two bin that at the two sides of the line.
        '''
        Graph = cp.deepcopy(Graph)
        Graph[bin1] = np.append(Graph[bin1], bin2)
        Graph[bin2] = np.append(Graph[bin2], bin1)
        return Graph


class WindowsBin(Bin):
    def __init__(self, 
                 xbin: int, 
                 ybin: int, 
                 x: int, 
                 y: int, 
                 calculator: WindowsCalculator = None
                ) -> None:
        super().__init__(xbin, ybin, x, y)

        self._state = 1
        # 0 -> (This bin is) selected to be abandoned; 
        # 1 -> (This bin is) kept as a part of the environment.
        self._calculator = calculator
        self._time = time.time()
    
    def state_change(self, 
                     batch: pyglet.graphics.Batch,
                     occu_map: np.ndarray,
                     **kwargs
                    ) -> np.ndarray:
        if time.time() - self._time >= 0.5:
            self._state = 1 - self._state
            if self._state == 0:
                self._plot_diagonal(batch=batch, **kwargs)
                occu_map[self.id-1] = np.nan
            elif self._state == 1:
                self._erase_diagonal()
                occu_map[self.id-1] = 0
            self._time = time.time()
            return occu_map
        else:
            return occu_map
        
    def _plot_diagonal(self, batch: pyglet.graphics.Batch, **kwargs):
        x1, y1 = self._calculator.cor_to_scr(self.bl[0], self.bl[1])
        x2, y2 = self._calculator.cor_to_scr(self.ur[0], self.ur[1])
        self.Line1 = shapes.Line(x1, y1, x2, y2, batch = batch, color = SELECTED_COLOR, **kwargs)

        x1, y1 = self._calculator.cor_to_scr(self.br[0], self.br[1])
        x2, y2 = self._calculator.cor_to_scr(self.ul[0], self.ul[1])
        self.Line2 = shapes.Line(x1, y1, x2, y2, batch = batch, color = SELECTED_COLOR, **kwargs)

    def _erase_diagonal(self):
        self.Line1.delete()
>>>>>>> bebfe805e63d3226d6196e1b8ff4c78089a1de31
        self.Line2.delete()