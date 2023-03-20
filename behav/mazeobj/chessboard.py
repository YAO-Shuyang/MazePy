'''
Date: March 15th, 2023
Author: Shuyang Yao

To plot a chessboard on the GUI.
'''

import pyglet

import numpy as np

from . import HEIGHT, WIDTH
from mazepy.behav.grid import GridBasic
from mazepy.behav.transloc import idx_to_loc
from .windowobj import WindowsBin, WindowsWall, WindowsCalculator


class ChessBoard(GridBasic):
    def __init__(self,
                 xbin: int,
                 ybin: int, 
                 aspect: str or float = 'equal'
                ) -> None:
        '''
        Parameter
        ---------
        aspect: str or float, optional
            Values in {'equal', 'auto', [ANY FLOAT]]} are available. 'equal' means 
              each bin has a equal and squared shape, while 'auto' means the shape 
              of each bin should obey the shape of the window (most likely, has a 
              ratio of 16:9). Besides, you can also input a float value, a ratio 
              (height/width), to set the shape of each bin.

        Note
        ----
        To generate chessboard-like grid pattern on the GUI.
        '''
        super().__init__(xbin = xbin, ybin = ybin)
        self._aspect = aspect
        self._calc_place_area()
        self._generate_batch()

    def _calc_place_area(self):
        f'''
        Calculate the basic arguments of the area to place the chess board: the (x,y) location of the four corners. 

        Note that HEIGHT = {HEIGHT}, WIDTH = {WIDTH}
        '''
        if self._aspect == 'equal':
            coef = 1
        elif self._aspect == 'auto':
            coef = self.xbin/self.ybin * HEIGHT/WIDTH
        else:
            if type(self._aspect) not in [float, int]:
                raise TypeError(f"{self._aspect} is an invalid value for aspect.")
            coef = self._aspect

        if self.ybin/self.xbin >= HEIGHT/WIDTH:
            board_width = (0.9 - 0.1)*HEIGHT/self.ybin*self.xbin / coef
            top = 0.9*HEIGHT
            bot = 0.1*HEIGHT
            lef = WIDTH/2 - board_width*0.5
            rig = WIDTH/2 + board_width*0.5
        else:
            board_height = (0.9-0.1)*WIDTH/self.xbin*self.ybin * coef
            top = HEIGHT/2 + board_height*0.5
            bot = HEIGHT/2 - board_height*0.5
            lef = 0.1*WIDTH
            rig = 0.9*WIDTH
        
        self.wcalc = WindowsCalculator(xbin=self.xbin, ybin=self.ybin, 
                                       ur_val=(rig, top), ul_val=(lef, top),
                                       br_val=(rig, bot), bl_val=(lef, bot))
    
    def _generate_batch(self):
        self.batch = pyglet.graphics.Batch()

    def create_chessboard_edge(self, **kwargs) -> None:
        # horizontal
        h = []
        for x in range(self.xbin):
            l = []
            for y in range(self.ybin+1):
                Edge = WindowsWall(self.xbin, self.ybin, x=x, y=y, calculator=self.wcalc, dirc='h')
                Line = Edge.plot_line_on_batch(batch=self.batch, **kwargs)
                l.append(Edge)
            h.append(l)

        # vertical
        v = []
        for x in range(self.xbin+1):
            l = []
            for y in range(self.ybin):
                Edge = WindowsWall(self.xbin, self.ybin, x=x, y=y, calculator=self.wcalc, dirc='v')
                Line = Edge.plot_line_on_batch(batch=self.batch, **kwargs)
                l.append(Edge)
            v.append(l)
        
        self.index = {'v':v, 'h':h}
    
    def create_chessboard_bin(self) -> None:
        self.Bins = []
        for i in range(self.xbin*self.ybin):
            x, y = idx_to_loc(i+1, self.xbin, self.ybin)
            node = WindowsBin(self.xbin, self.ybin, x=x, y=y, calculator=self.wcalc)
            self.Bins.append(node)