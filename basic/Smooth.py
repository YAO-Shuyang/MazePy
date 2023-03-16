"""
Provide Gaussian Smooth for complex environment. It can avoid the internal walls.
"""

import numpy as np
from mazepy.behav.gridbin import GridBasic
from mazepy.behav.graph import OpenFieldGraph
from mazepy.behav.transloc import loc_to_idx, idx_to_loc
import sklearn.preprocessing


class GaussianSmooth(GridBasic):
    def __init__(self, 
                 xbin: int, 
                 ybin: int, 
                 Graph: dict = None
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
        self.Graph = Graph
        
    def _Gaussian(self, x=0, sigma=2, pi=3.1416, ):
        x = x * (48 / self.)
        return 1 / (sigma * np.sqrt(pi * 2)) * np.exp(- x * x / (sigma * sigma * 2))

    def _Cartesian_distance(curr, surr, nx = 48):
        # curr is the node id
        curr_x, curr_y = idx_to_loc(curr, nx = nx, ny = nx)
        surr_x, surr_y = idx_to_loc(surr, nx = nx, ny = nx)
        return np.sqrt((curr_x - surr_x)*(curr_x - surr_x)+(curr_y - surr_y)*(curr_y - surr_y))

    def SmoothMatrix(maze_type = 1, sigma = 2, _range = 7, nx = 48):
    
        smooth_matrix = np.zeros((nx*nx,nx*nx), dtype = np.float64)
    
        for curr in range(1,nx*nx+1):
            SurrMap = {}
            SurrMap[0]=[curr]
            Area = [curr]
    
            step = int(_range * 1.5)
            smooth_matrix[curr-1,curr-1] = Gaussian(0,sigma = sigma, nx = nx)
            for k in range(1,step+1):
                SurrMap[k] = np.array([],dtype = np.int32)
                for s in SurrMap[k-1]:
                    for j in range(len(graph[s])):
                        length = Cartesian_distance(curr, graph[s][j], nx = nx)
                        if graph[s][j] not in Area and length <= _range:
                            Area.append(graph[s][j])
                            SurrMap[k] = np.append(SurrMap[k], graph[s][j])
                            smooth_matrix[curr-1, graph[s][j]-1] = Gaussian(length,sigma = sigma, nx = nx)

        smooth_matrix = sklearn.preprocessing.normalize(smooth_matrix, norm = 'l1')
        return smooth_matrix