import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import numpy as np
import warnings

from mazepy.behav.grid import GridBasic
from mazepy.behav.transloc import loc_to_idx, idx_to_loc
from mazepy.behav.graph import WallMatrix

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
                 occu_map: np.ndarray = None,
                ) -> None:
        """
        Parameter
        ---------
        Graph: dict, required
            Each mouse event will probably change the state of a certain edge which will cause
              a change of the connection between the two bins that on the two side of the edge.
            The graph of the environment contrains the connections between each two nearby bins
              in the environment.
        occu_map: np.ndarray object, optional
            To keep some of the area empty.
        save_loc: str, optional
            If it is not None, it should be a valid directory that to save the figure.
        """
        super().__init__(xbin = xbin, ybin = ybin)

        if occu_map is None:
            occu_map = np.zeros(xbin*ybin)

        if len(Graph.keys()) != xbin*ybin:
            raise IndexError(f"""The Graph has {Graph.keys()} keys and thus doesn't suit the 
                             environment which has {xbin*ybin} bins. The number of keys must be
                             matched with the number of bins.""")
        if occu_map.shape[0] != self.xbin*self.ybin:
            raise IndexError(f"""{occu_map.shape} does not fit the bin number of the 
                             environment. It should be 1d and has a shape of 
                             ({self.xbin*self.ybin},).""")
        self.Graph = Graph
        self.occu_map = occu_map

    def _WallMatrix(self):
        self._vertical_walls, self._horizont_walls = WallMatrix(self.Graph, self.xbin, self.ybin)
        return self._vertical_walls, self._horizont_walls

    @property
    def vertical_wall(self):
        return self._vertical_walls
    
    @property
    def horizont_wall(self):
        return self._horizont_walls

    def DrawMazeProfile(self, 
                        ax: Axes = None,
                        figsize: tuple = (6,6),
                        rate_map: np.ndarray = None,
                        imshow_kwgs: dict = {},
                        **kwargs,
                       ) -> Axes:
        """
        Parameter
        ---------
        ax: Axes object, optional
            If it is None, it will make up automatically.
        rate_map: np.ndarray, optional
            Note that the rate map must have the same length as the environment, that
              is, xbin*ybin, or it will raise an error. rate map should be 1d.
        save_loc: str, optional
            If it is not None, it should be a valid directory that to save the figure.
        """
        if rate_map is not None:
            if rate_map.shape[0] != self.xbin*self.ybin:
                raise IndexError(f"""{rate_map.shape} does not fit the bin number of the 
                                 environment. It should be 1d and has a shape of 
                                 ({self.xbin*self.ybin},).""")

        self._WallMatrix()
        v, h = self._vertical_walls, self._horizont_walls

        if ax is None:
            fig = plt.figure(figsize = figsize)
            ax = plt.axes()
            ax.set_aspect('equal')

        if rate_map is None and self.occu_map is not None:
            ax.imshow(np.reshape(self.occu_map, [self.xbin, self.ybin]), **imshow_kwgs)
        elif rate_map is not None and self.occu_map is not None:
            rate_map_modi = rate_map + self.occu_map
            ax.imshow(np.reshape(rate_map_modi, [self.xbin, self.ybin]), **imshow_kwgs)
        elif rate_map is not None and self.occu_map is None:
            ax.imshow(np.reshape(rate_map, [self.xbin, self.ybin]), **imshow_kwgs)
            warnings.warn("Note that we do not recommend you to draw rate map without occu_map.")

        ax.invert_yaxis()

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if v[i,j] == 1:
                    ax.plot([j-0.5, j-0.5],[i-0.5,(i+1)-0.5], **kwargs)

        for j in range(h.shape[0]):
            for i in range(h.shape[1]):
                if h[j,i] == 1:
                    ax.plot([i-0.5, (i+1)-0.5],[j-0.5,j-0.5], **kwargs)

        return ax