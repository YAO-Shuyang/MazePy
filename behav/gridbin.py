'''
Date: March 14th, 2023
Author: Shuyang Yao

Purpose: 
It could be used to divided recording data space into bins. The parameters can be purely the location of particular animal, 
or a combination of different variables like time and location.
'''

import numpy as np
import scipy.stats

from mazepy.behav.transloc import pvl_to_idx
from mazepy.behav.grid import GridSize
from mazepy.gui import ENV_OUTSIDE_SHAPE

class GridBin(GridSize):
    '''
    Parameter
    ---------
    xmax: float, required
        The top limit of recorded data on dimension x.
    ymax: float, required
        The top limit of recorded data on dimension x.
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    '''
    # We only accept dimension n = 2 to divide variables into different bins.
    def __init__(self, xbin: int, ybin: int, xmax: float, ymax: float, 
                 xmin: int or float = 0, ymin: int or float = 0) -> None:
        super().__init__(xbin, ybin, xmax, ymax, xmin, ymin)

    def pre_process(self, MAT):
        '''
        Note
        ----
        Pre-process the matrix to avoid overflow.
        '''
        self.xmax = self.xmax - self.xmin
        self.ymax = self.ymax - self.ymin
        MAT[0, :] = MAT[0, :] - self.xmin
        MAT[1, :] = MAT[1, :] - self.ymin

        MAT[0, np.where(MAT[0, :] >= self.xmax-0.0001)[0]] = self.xmax - 0.0001 
        MAT[1, np.where(MAT[1, :] >= self.ymax-0.0001)[0]] = self.ymax - 0.0001
        self.MAT = MAT

    @staticmethod
    def divide_space(
        behav_pos: np.ndarray, 
        xbin: int, ybin: int, 
        xmax: float, 
        ymax: float, 
        xmin: int or float = 0, 
        ymin: int or float = 0, 
    ) -> np.ndarray:
        """
        divide_space: this function is used to divide the value space into grid-like bins and get the bin ID 
                      of each value point (recorded by each frame).

        Parameters
        ----------
        behav_pos : np.ndarray
            The behavior position. It usually has a shape of (n, T), where n represent the dimension (default: 2) and T 
            represents the total number of frames that recorded.
        xbin : int
            The number of bins at axis x
        ybin : int
            The number of bins at axis y
        xmax : float
            The maximum x value
        ymax : float
            The maximum y value
        xmin : int or float, optional
            The minimum x value, by default 0
        ymin : int or float, optional
            The minimum y value, by default 0

        Returns
        -------
        np.ndarray
            1d numpy.ndarray object, with a length of T.
        """
        obj = GridBin(xbin=xbin, ybin=ybin, xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin)
        obj.pre_process(behav_pos)
        behav_nodes = pvl_to_idx(x = behav_pos[0, :], y = behav_pos[1, :], xbin = obj.xbin, ybin = obj.ybin, 
                                 xmax = obj.xmax, ymax = obj.ymax)
        return behav_nodes
    
    @staticmethod
    def calc_rate_map(
        behav_time: np.ndarray, 
        behav_pos: np.ndarray, 
        xbin: int, ybin: int, 
        xmax: float, 
        ymax: float, 
        xmin: int or float = 0, 
        ymin: int or float = 0, 
        fps: float = 30.
    ) -> np.ndarray:
        """
        calc_rate_map: calculate the behavior rate map

        Parameters
        ----------
        behav_time : np.ndarray
            The behavior timestamp
        behav_pos : np.ndarray
            The behavior position
        xbin : int
            The number of bins at axis x
        ybin : int
            The number of bins at axis y
        xmax : float
            The maximum x value
        ymax : float
            The maximum y value
        xmin : int or float, optional
            The minimum x value, by default 0
        ymin : int or float, optional
            The minimum y value, by default 0
        fps : float, optional
            Frame per seconde, by default 30.

        Returns
        -------
        np.ndarray
            The occupation time at each bin.
        """
        frame_interval = np.append(np.ediff1d(behav_time), fps)

        behav_nodes = GridBin.divide_space(behav_pos, xbin=xbin, ybin=ybin, xmax=xmax, ymax=ymax, xmin=xmin, ymin=ymin)
        
        _nbins = xbin * ybin
        _coords_range = [0, _nbins +0.0001 ]

        occu_time = scipy.stats.binned_statistic(
            behav_nodes,
            frame_interval,
            bins=_nbins,
            statistic="sum",
            range=_coords_range
        )
        occu_time[np.where(occu_time==0)[0]] = np.nan
        return occu_time
    

if __name__ == '__main__':
    # test
    import pickle
    import copy as cp

    with open('trace_behav.pkl', 'rb') as handle:
        trace = pickle.load(handle) # trace is a dict object
    
    MAT = cp.deepcopy(trace['processed_pos_new']).T
    behav_time = cp.deepcopy(trace['behav_time'])
    behav_nodes = cp.deepcopy(trace['behav_nodes'])  # Answer

    GB = GridBin(xmax=960, ymax=960, xbin=48, ybin=48)
    GB.divide_space(MAT)
    GB.cal_rate_map(behav_time)

    # Test if the new-generated behav_nodes is correct.
    print(np.where((GB.behav_nodes - behav_nodes)!= 0))
