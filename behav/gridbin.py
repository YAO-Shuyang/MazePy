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
from dataclasses import dataclass


@dataclass
class GridBasic(object):
    '''
    Parameter
    ---------
    xbin: int, optional
        The total bin number of dimension x
        default: 10
    ybin: int, optional
        The total bin number of dimension y
        default: 10
    '''
    xbin: int
    ybin: int

@dataclass
class GridSize(GridBasic):
    '''
    Note
    ----
    The basic parameter to bin the value space into grid-like bins.

    Parameter
    ---------
    xmax: float, required
        The top limit of recorded data on dimension x.
    ymax: float, required
        The top limit of recorded data on dimension y.
    xmin: float, required
        The bottom limit of recorded data on dimension x.
    ymin: float, required
        The bottom limit of recorded data on dimension y.
    '''
    xmax: float
    ymax: float
    xmin: float = 0
    ymin: float = 0

@dataclass
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

    def _pre_process(self, MAT):
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

    def divide_space(self, MAT):
        '''
        Parameter
        ---------
        MAT: numpy.ndarray object, required
            It usually has a shape of (n, T), where n represent the dimension (default: 2) and T 
            represents the total number of frames that recorded.

        Note
        ----
        This function is used to divide the value space into grid-like bins and get the bin ID 
          of each value point (recorded by each frame).

        Return
        ------
        1d numpy.ndarray object, with a length of T.
        '''
        self._pre_process(MAT = MAT)
        self.behav_nodes = pvl_to_idx(prec_value_loc = self.MAT, xbin = self.xbin, ybin = self.ybin, 
                                      xmax = self.xmax, ymax = self.ymax)
        return self.behav_nodes

    def cal_rate_map(self, behav_time:np.ndarray):
        '''
        Parameter
        ---------
        behav_time: numpy.ndarray object, required
            It is usually a 1d vector with a shape of (T,).

        Note
        ----
        This function is used to calculate the rate map of behavior. For this purpose, 
          the time stamp of related behavior data is required.
        
        '''
        frame_interval = np.append(np.ediff1d(behav_time), 0)
        
        _nbins = self.xbin * self.ybin
        _coords_range = [0, _nbins +0.0001 ]

        self.occu_time = scipy.stats.binned_statistic(self.behav_nodes,
                                                      frame_interval,
                                                      bins=_nbins,
                                                      statistic="sum",
                                                      range=_coords_range)
        return self.occu_time
    

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
