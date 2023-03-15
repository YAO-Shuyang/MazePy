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

from mazepy.behav.gridbin import GridBasic
from mazepy.behav.transloc import loc_to_idx

UNSELECTED_COLOR = (204, 255, 204)
SELECTED_COLOR = (0, 0, 0)

class Edge(GridBasic):
    def __init__(self, 
                 xbin: int,
                 ybin: int,
                 row_val: int,
                 col_val: int,
                 dirc: str = 'v',
                ) -> None:
        '''
        Parameter
        ---------
        xbin: int, required
            The total bin number of dimension x
        ybin: int, required
            The total bin number of dimension y
        row_val: int, required
            The row that the edge lays on.
        col_val: int, required
            The column that the edge lays on.
        direction: str, optional
            Values in {'v', 'h'} are available. 'v' represents vertical edges, while 'h' represents
              horizontal ones.
        '''
        super().__init__(xbin = xbin, ybin = ybin)

        if dirc in ['v','h']:
            self._direction = dirc
        else:
            raise ValueError(f"{dirc} is an invalid value for direction.")
        
        self._row = row_val
        self._col = col_val
        self._state = 0   # the state of the edge: 0 -> no wall; 1 -> has a wall.

    @property
    def direction(self):
        return self._direction
    
    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

class MazeInnerWall(Edge):
    '''
    Note
    ----
    Define the properties of internal structure
    '''
    def __init__(self, 
                 xbin: int,
                 ybin: int,
                 row_val: int,
                 col_val: int,
                 four_corner: dict,
                 dirc: str = 'v'
                ) -> None:
        '''
        Parameter
        ---------
        xbin: int, required
            The total bin number of dimension x
        ybin: int, required
            The total bin number of dimension y
        row_val: int, required
            The row that the edge lays on.
        col_val: int, required
            The column that the edge lays on.
        four_corner: dict, required
            Class ChessBoard in /behav/mazeobj/chessboard.py will generate this dict, such as:
                            {'upper left': np.array([lef, top]),
                            'upper right': np.array([rig, top]),
                            'bottom left': np.array([lef, bot]),
                            'bottom right': np.array([rig, bot])}
        dirc: str, optional
            Values in {'v', 'h'} are available. 'v' represents vertical edges, while 'h' represents
              horizontal ones.
        '''
        super().__init__(xbin = xbin, ybin = ybin, row_val = row_val, col_val = col_val, dirc = dirc)
        self.four_corner = four_corner
        self._color = UNSELECTED_COLOR
        self._calc_two_ends_xy()
        self._find_two_bins_id()
        self.time = time.time()

    def _find_two_bins_id(self):
        '''
        Find the two bins that on the two side of the edge.
        '''
        if self._direction == 'v':
            self.bin1 = loc_to_idx(cell_x = self._col-1, cell_y = self._row, xbin = self.xbin) if self._col >= 1 else None
            self.bin2 = loc_to_idx(cell_x = self._col, cell_y = self._row, xbin = self.xbin) if self._col <= self.xbin-1 else None

    def _calc_two_ends_xy(self):
        '''
        Calculate the (x,y) location of the two ends of the edge.
        '''
        width = self.four_corner['upper right'][0] - self.four_corner['upper left'][0]
        height = self.four_corner['upper left'][1] - self.four_corner['bottom left'][1]

        if self._direction == 'v':
            x_top = width * self._col / self.xbin + self.four_corner['upper left'][0]
            x_bot = x_top
            y_top = height * (self._row+1) / self.ybin + self.four_corner['bottom left'][1]
            y_bot = height * self._row / self.ybin + self.four_corner['bottom left'][1]
        
            x1, y1, x2, y2 = x_top, y_top, x_bot, y_bot

        elif self._direction == 'h':
            x_lef = width * self._col / self.xbin + self.four_corner['upper left'][0]
            x_rig = width * (self._col+1) / self.xbin + self.four_corner['upper left'][0]
            y_lef = height * self._row / self.ybin + self.four_corner['bottom left'][1]
            y_rig = y_lef

            x1, y1, x2, y2 = x_lef, y_lef, x_rig, y_rig
        
        self._x1 = x1
        self._x2 = x2
        self._y1 = y1
        self._y2 = y2

    def plot_line_on_batch(self, 
                           batch: pyglet.graphics.Batch
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
        line = shapes.Line(self._x1, self._y1, self._x2, self._y2, width = 5, color = self._color, batch = batch)
        line.opacity = 255
        self.line = line
        return line

    def state_change(self, 
                     batch: pyglet.graphics.Batch,
                     Graph: dict = {}
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
        A dict
        '''
        if time.time() - self.time >= 0.5:
            self._state = 1 - self._state
            self._color = UNSELECTED_COLOR if self._state == 0 else SELECTED_COLOR
            self.time = time.time()
            return self.plot_line_on_batch(batch = batch)
    