<<<<<<< HEAD
"""
Provide Gaussian Smooth for complex environment. It can avoid the internal walls.
"""

import numpy as np
from mazepy.behav.grid import GridBasic
from mazepy.behav.graph import OpenFieldGraph, Graph
from mazepy.behav.transloc import loc_to_idx, idx_to_loc
import sklearn.preprocessing


class GaussianSmooth(GridBasic):
    def __init__(self, 
                 xbin: int, 
                 ybin: int, 
                 G: dict = None
                ) -> None:
        """
        Parameter
        ---------
        xbin: int, required
            Defines the total number of bins at dimension x.
        ybin: int, required
            Defines the total number of bins at dimension y.
        Graph: dict, optional
            Default - None, if it is None, a graph for open field will be generated.
        """
        super().__init__(xbin = xbin, ybin = ybin)
        if G is None:
            G = OpenFieldGraph(xbin=xbin, ybin=ybin)

        self.Graph = G
        self.G = Graph(Graph=G, xbin=xbin, ybin=ybin)
        
    def _Gaussian(self, x: float = 0, sigma: float = 2, pi: float = 3.1416):
        x = x * self.unit
        return 1 / (sigma*np.sqrt(pi*2)) * np.exp(- x*x / (sigma**2 * 2))

    def _Cartesian(self, bin1: int, bin2: int):
        """
        Parameter
        ---------
        bin1, bin2: int, required
            The bin ID of the two bins, respectively.
        """
        x1, y1 = idx_to_loc(bin1, xbin=self.xbin, ybin=self.ybin)
        x2, y2 = idx_to_loc(bin2, xbin=self.xbin, ybin=self.ybin)
        return np.sqrt((x1 - x2)**2 + (y1- y2)**2)
    
    def _maze_distance(self, bin1: int, bin2: int):
        x1, y1 = idx_to_loc(bin1, xbin=self.xbin, ybin=self.ybin)
        x2, y2 = idx_to_loc(bin2, xbin=self.xbin, ybin=self.ybin)
        return self.G.shortest_distance((x1, y1), (x2, y2))        

    def SmoothMatrix(self, sigma: float = 2, mode: str = 'cartesian'):
        """
        Parameter
        ---------
        sigma: float, optional
            The most important argument of Gaussian distribution. It determines the range of 
              smooth. The bigger the sigma is, the larger the smooth window will be, and thus
              the more smooth the data will be.
        mode: str, optional
            It determines how to calculate distance in maze. There are currently 2 modes to be
              chosen: 'cartesian' and 'maze_distance'. If you choose 'cartesian', it will
              calculate the cartesian distance of pairs of points. If you choose 'maze_distance',
              it would return the shortest path between two points in the maze.
            Note that the maze distance could be affected by how to place the walls (the shortest
              path should not cross the wall, unlike what the cartesian distance usually does)
            Note that these 2 modes are equivalent in the open field, for there's no wall in it.
              
        Note
        ----
        Calculate the smooth matrix to smooth rate map.
        """
        if mode not in ['cartesian', 'maze_distance']:
            raise ValueError(f"Only 'cartesian' and 'maze_distance' are valid values for parameter 'mode', instead of {mode}.")
        
        smooth_matrix = np.zeros((self.xbin*self.ybin, self.xbin*self.ybin), dtype = np.float64)
        dist_func = self._Cartesian if mode == 'cartesian' else self._maze_distance

        step = 100
        thre = 0.005  
        """
        step: Smooth stack level.
        thre: Smooth threshold. If the contribution of bin1 to another bin (bin2) is 
        """ 
                
        for curr in range(1, self.xbin*self.ybin + 1):
            SurrMap = {}
            SurrMap[0]=[curr]
            Area = [curr]
    
            smooth_matrix[curr-1,curr-1] = self._Gaussian(0, sigma = sigma)
            for k in range(1,step+1):
                SurrMap[k] = np.array([],dtype = np.int32)
                for s in SurrMap[k-1]:
                    for j in range(len(self.Graph[s])):
                        dis = dist_func(curr, self.Graph[s][j])
                        val = self._Gaussian(dis,sigma = sigma)

                        if self.Graph[s][j] not in Area and val >= thre:
                            Area.append(self.Graph[s][j])
                            SurrMap[k] = np.append(SurrMap[k], self.Graph[s][j])
                            smooth_matrix[curr-1, self.Graph[s][j]-1] = val

                if val < thre:
                    break

        smooth_matrix = sklearn.preprocessing.normalize(smooth_matrix, norm = 'l1')
=======
"""
Provide Gaussian Smooth for complex environment. It can avoid the internal walls.
"""

import numpy as np
from mazepy.behav.grid import GridBasic
from mazepy.behav.graph import OpenFieldGraph, Graph
from mazepy.behav.transloc import loc_to_idx, idx_to_loc
import sklearn.preprocessing


class GaussianSmooth(GridBasic):
    def __init__(self, 
                 xbin: int, 
                 ybin: int, 
                 G: dict = None
                ) -> None:
        """
        Parameter
        ---------
        xbin: int, required
            Defines the total number of bins at dimension x.
        ybin: int, required
            Defines the total number of bins at dimension y.
        Graph: dict, optional
            Default - None, if it is None, a graph for open field will be generated.
        """
        super().__init__(xbin = xbin, ybin = ybin)
        if G is None:
            G = OpenFieldGraph(xbin=xbin, ybin=ybin)

        self.Graph = G
        self.G = Graph(Graph=G, xbin=xbin, ybin=ybin)
        
    def _Gaussian(self, x: float = 0, sigma: float = 2, pi: float = 3.1416):
        x = x * self.unit
        return 1 / (sigma*np.sqrt(pi*2)) * np.exp(- x*x / (sigma**2 * 2))

    def _Cartesian(self, bin1: int, bin2: int):
        """
        Parameter
        ---------
        bin1, bin2: int, required
            The bin ID of the two bins, respectively.
        """
        x1, y1 = idx_to_loc(bin1, xbin=self.xbin, ybin=self.ybin)
        x2, y2 = idx_to_loc(bin2, xbin=self.xbin, ybin=self.ybin)
        return np.sqrt((x1 - x2)**2 + (y1- y2)**2)
    
    def _maze_distance(self, bin1: int, bin2: int):
        x1, y1 = idx_to_loc(bin1, xbin=self.xbin, ybin=self.ybin)
        x2, y2 = idx_to_loc(bin2, xbin=self.xbin, ybin=self.ybin)
        return self.G.shortest_distance((x1, y1), (x2, y2))        

    def SmoothMatrix(self, sigma: float = 2, mode: str = 'cartesian'):
        """
        Parameter
        ---------
        sigma: float, optional
            The most important argument of Gaussian distribution. It determines the range of 
              smooth. The bigger the sigma is, the larger the smooth window will be, and thus
              the more smooth the data will be.
        mode: str, optional
            It determines how to calculate distance in maze. There are currently 2 modes to be
              chosen: 'cartesian' and 'maze_distance'. If you choose 'cartesian', it will
              calculate the cartesian distance of pairs of points. If you choose 'maze_distance',
              it would return the shortest path between two points in the maze.
            Note that the maze distance could be affected by how to place the walls (the shortest
              path should not cross the wall, unlike what the cartesian distance usually does)
            Note that these 2 modes are equivalent in the open field, for there's no wall in it.
              
        Note
        ----
        Calculate the smooth matrix to smooth rate map.
        """
        if mode not in ['cartesian', 'maze_distance']:
            raise ValueError(f"Only 'cartesian' and 'maze_distance' are valid values for parameter 'mode', instead of {mode}.")
        
        smooth_matrix = np.zeros((self.xbin*self.ybin, self.xbin*self.ybin), dtype = np.float64)
        dist_func = self._Cartesian if mode == 'cartesian' else self._maze_distance

        step = 100
        thre = 0.005  
        """
        step: Smooth stack level.
        thre: Smooth threshold. If the contribution of bin1 to another bin (bin2) is 
        """ 
                
        for curr in range(1, self.xbin*self.ybin + 1):
            SurrMap = {}
            SurrMap[0]=[curr]
            Area = [curr]
    
            smooth_matrix[curr-1,curr-1] = self._Gaussian(0, sigma = sigma)
            for k in range(1,step+1):
                SurrMap[k] = np.array([],dtype = np.int32)
                for s in SurrMap[k-1]:
                    for j in range(len(self.Graph[s])):
                        dis = dist_func(curr, self.Graph[s][j])
                        val = self._Gaussian(dis,sigma = sigma)

                        if self.Graph[s][j] not in Area and val >= thre:
                            Area.append(self.Graph[s][j])
                            SurrMap[k] = np.append(SurrMap[k], self.Graph[s][j])
                            smooth_matrix[curr-1, self.Graph[s][j]-1] = val

                if val < thre:
                    break

        smooth_matrix = sklearn.preprocessing.normalize(smooth_matrix, norm = 'l1')
>>>>>>> bebfe805e63d3226d6196e1b8ff4c78089a1de31
        return smooth_matrix