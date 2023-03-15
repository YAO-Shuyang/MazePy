import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import numpy as np

from mazepy.behav.gridbin import GridBasic
from mazepy.behav.transloc import loc_to_idx, idx_to_loc

# plot environment profile
class MazeProfile(GridBasic):
    '''
    This class is generate to plot the maze profile (the environment with internal.
      structure, if it has).
    '''
    def __init__(self,
                 xbin: int,
                 ybin: int,
                 Graph: dict = {},
                 save_loc: str = None
                ) -> None:
        """
        Parameter
        ---------
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.
            The graph of the environment contrains the connections between each two nearby bins
              in the environment.
        save_loc: str, optional
            If it is not None, it should be a valid directory that to save the figure.
        """
        super().__init__(xbin = xbin, ybin = ybin)
        self.Graph = Graph
        self._loc = save_loc
        axes = self.DrawMazeProfile(color = 'black', linewidth = 2)
        if save_loc is None:
            plt.show()
        else:
            plt.savefig(save_loc+'.png', dpi = 600)
            plt.savefig(save_loc+'.svg', dpi = 600)
            plt.close()

    def _WallMatrix(self) -> None:
        """
        Return
        ------
        numpy.ndarray Matrix * 2
        Wall matrix will return 2 matrix: vertical_walls represents verticle walls which has a 
          shape of (xbin+1, ybin) and horizont_walls with a shape of (xbin, ybin+1).
        """
        self.vertical_walls = np.ones((self.ybin, self.xbin+1), dtype = np.int64)
        self.horizont_walls = np.ones((self.ybin+1, self.xbin), dtype = np.int64)

        for i in range(1, self.xbin*self.ybin):
            x, y = idx_to_loc(i, xbin=self.xbin)

            surr = self.Graph[i]
            for s in surr:
                if s == i + 1:
                    self.vertical_walls[y, x+1] = 0
                elif s == i - 1:
                    self.vertical_walls[y, x] = 0
                elif s == i + self.xbin:
                    self.horizont_walls[y+1, x] = 0
                elif s == i - self.xbin:
                    self.horizont_walls[y, x] = 0
                else:
                    raise ValueError(f"Bin {s} is not connected with Bin {i}.")

    def DrawMazeProfile(self, 
                        axes: Axes = None,
                        **kwargs
                       ) -> Axes:
        """
        Parameter
        ---------
        axes: Axes object, optional
            If it is None, it will make up automatically.
        save_loc: str, optional
            If it is not None, it should be a valid directory that to save the figure.
        """
        self._WallMatrix()
        v, h = self.vertical_walls, self.horizont_walls

        if axes is None:
            fig = plt.figure(figsize = (6,6))
            axes = plt.axes()
            axes.set_aspect('equal')

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if v[i,j] == 1:
                    axes.plot([j-0.5, j-0.5],[i-0.5,(i+1)-0.5], **kwargs)

        for j in range(h.shape[0]):
            for i in range(h.shape[1]):
                if h[j,i] == 1:
                    axes.plot([i-0.5, (i+1)-0.5],[j-0.5,j-0.5], **kwargs)
        return axes