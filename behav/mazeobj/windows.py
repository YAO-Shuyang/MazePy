import pyglet
from pyglet.window import key, mouse
from pyglet import shapes

import numpy as np
import copy as cp
import time

from mazepy.behav.mazeobj.guiasset import BACKGROUND_IMG
from mazepy.behav.mazeobj.chessboard import ChessBoard
from mazepy.behav.mazeobj import WINDOW
from mazepy.behav.gridbin import GridBasic
from mazepy.behav.transloc import pvl_to_edge, pvl_to_loc, pvl_to_idx


MOUSE_STATE = {'x': None, 'y': None, 'button': None, 'modifiers': None}
KEY_STATE = set()

class Background(pyglet.sprite.Sprite):
    def __init__(self, *args, **kwargs):
        img = BACKGROUND_IMG
        img.width = WINDOW.width
        img.height = WINDOW.height
        img.anchor_x = img.width // 2
        img.anchor_y = img.height // 2
        super().__init__(
            img, x=WINDOW.width // 2, y=WINDOW.height // 2, *args, **kwargs
        )


class MainWindow(GridBasic):
    def __init__(self,
                 xbin: int,
                 ybin: int,
                 Graph: dict,
                 occu_map: np.ndarray, 
                 aspect: str or float = 'equal',
                 **kwargs
                ) -> None:
        ''''
        Parameter
        ---------
        xbin: int, required
            The total bin number of dimension x
        ybin: int, required
            The total bin number of dimension y
        Graph: dict, required
            The graph of the picture
        occu_map: np.ndarray 1d vector, optional, 
            Occupation map contains only 0 and np.nan.
            0 -> this bin will posibly be covered by animals; 1 -> this bin will never be 
              covered by animals (for it is even not a part of the environment.)
        aspect: str, {'equal', 'auto', float}
            The shape of the bin that projected on the GUI.
        '''
        super().__init__(xbin = xbin, ybin = ybin)
        self.bg = Background()
        self.Graph = cp.deepcopy(Graph)
        self.occu_map = cp.deepcopy(occu_map)
        self.cb = ChessBoard(xbin = xbin, ybin = ybin, aspect = aspect)
        self.cb.create_chessboard_edge(**kwargs)
        self.cb.create_chessboard_bin()
        self.keys = kwargs

    @WINDOW.event
    def on_mouse_press(x, y, button, modifiers):
        MOUSE_STATE['x'] = x
        MOUSE_STATE['y'] = y
        MOUSE_STATE['button'] = button
        MOUSE_STATE['modifiers'] = modifiers

    @WINDOW.event
    def on_mouse_release(x, y, button, modifiers):
        MOUSE_STATE['x'] = None
        MOUSE_STATE['y'] = None
        MOUSE_STATE['button'] = None
        MOUSE_STATE['modifiers'] = None

    @WINDOW.event
    def on_key_press(symbol, modifiers):
        if symbol in [key.ENTER, key.RETURN]:
            WINDOW.close()

    def _is_out_of_range(self, x, y):
        return x >= self.cb.wcalc.br[0] or x <= self.cb.wcalc.bl[0] or y >= self.cb.wcalc.ul[1] or y <= self.cb.wcalc.bl[1]

    def _edge_state_change(self, x, y) -> None:
        if self._is_out_of_range(x, y):
            return
        
        dirc, xb, yb = pvl_to_edge(prec_value_loc = np.array([x - self.cb.wcalc.bl[0], y - self.cb.wcalc.bl[1]]), 
                                 xmax = self.cb.wcalc.xrange, ymax = self.cb.wcalc.yrange, xbin = self.xbin, ybin = self.ybin)
        nearest_edge = self.cb.index[dirc][xb][yb]
        self.Graph = nearest_edge.state_change(self.cb.batch, Graph = self.Graph, **self.keys)

    def _bin_state_change(self, x, y) -> None:
        if self._is_out_of_range(x, y):
            return

        idx = pvl_to_idx(prec_value_loc = np.array([x - self.cb.wcalc.bl[0], y - self.cb.wcalc.bl[1]]), 
                         xmax = self.cb.wcalc.xrange, ymax = self.cb.wcalc.yrange, xbin = self.xbin, ybin = self.ybin)
        self.occu_map = self.cb.Bins[idx-1].state_change(self.cb.batch, self.occu_map, **self.keys)

    def update(self, dt):
        if MOUSE_STATE['button'] == mouse.LEFT:
            self._edge_state_change(MOUSE_STATE['x'], MOUSE_STATE['y'])
        elif MOUSE_STATE['button'] == mouse.RIGHT:
            self._bin_state_change(MOUSE_STATE['x'], MOUSE_STATE['y'])
        self.draw()

    def draw(self):
        WINDOW.clear()
        self.bg.draw()
        self.cb.batch.draw()
