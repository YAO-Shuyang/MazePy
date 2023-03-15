import pyglet
from pyglet.window import key, mouse
from pyglet import shapes

import numpy as np
import time

from . import WINDOW
from .guiasset import BACKGROUND_IMG
from .chessboard import ChessBoard
from mazepy.behav.gridbin import GridBasic
from mazepy.behav.transloc import pvl_to_edge

MOUSE_STATE = {'x': None, 'y': None, 'button': None, 'modifiers': None}

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
                 aspect: str = 'equal'
                ) -> None:
        # # Init label
        super().__init__(xbin = xbin, ybin = ybin)
        self.bg = Background()
        self.cb = ChessBoard(xbin = xbin, ybin = ybin, aspect = aspect)
        self.cb.create_chessboard()

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

    def _state_change(self, x, y) -> None:
        if x >= self.cb.four_corner['bottom right'][0] or x <= self.cb.four_corner['bottom left'][0] or y >= self.cb.four_corner['upper left'][1] or y <= self.cb.four_corner['bottom left'][1]:
            return
        
        dirc, xb, yb = pvl_to_edge(prec_value_loc = np.array([x - self.cb.four_corner['bottom left'][0], 
                                                            y - self.cb.four_corner['bottom left'][1]]), 
                                 xmax = self.cb.four_corner['bottom right'][0] - self.cb.four_corner['bottom left'][0],
                                 ymax = self.cb.four_corner['upper left'][1] - self.cb.four_corner['bottom left'][1],
                                 xbin = self.xbin, ybin = self.ybin)
        nearest_edge = self.cb.index[dirc][xb][yb]
        line = nearest_edge.state_change(self.cb.batch)

    def update(self, dt):
        if MOUSE_STATE['button'] == mouse.LEFT:
            self._state_change(MOUSE_STATE['x'], MOUSE_STATE['y'])
        self.draw()

    def draw(self):
        WINDOW.clear()
        self.bg.draw()
        self.cb.batch.draw()