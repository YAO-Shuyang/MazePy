'''
Date: March 14th, 2023
Author: Shuyang Yao

Purpose: 
It could be used to divided recording parameters into bins. The parameters can be purely the location of particular animal, 
or a combination of different variables like time and location.
'''

try:
    import numpy as np
except:
    print("There's no package named 'numpy'.")
    assert False

try:
    import scipy
except:
    print("There's no package named 'scipy'.")
    assert False

from .transloc import pvl_to_idx

class Object():
    def __init__(self) -> None:
        pass

class BehavNode(Object):
    
    def __init__(self,
                 Mat: np.ndarray,
                 xmax: float|np.float64,
                 ymax: float|np.float64,
                 xbin: int = 10,
                 ybin: int = 10):
        '''
        Parameter
        ---------
        Mat: numpy.ndarray object, required
            It usually has a shape of (n, T), where n represent the dimension (default: 2) and 
            T represents the total number of frames that recorded.
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
        # We only accept dimension = 2 to divide variables into different bins.
        self.Mat = Mat
        self.xmax = xmax
        self.ymax = ymax
        self.xbin = xbin
        self.ybin = ybin

        # Divide precise value space into several bins
        self._divide_bin()

    def _divide_bin(self):
        '''
        Note
        ----
        This function is used to calculated the bin id of each frame.
        '''
        self.behav_nodes = pvl_to_idx(prec_value_loc = self.Mat, xbin = self.xbin, ybin = self.ybin, 
                                      xmax = self.xmax, ymax = self.ymax)

    def calculate_rate_map(self, behav_time:np.ndarray, fps:float|int = 30, time_unit:str = 'ms'):
        '''
        Parameter
        ---------
        behav_time: numpy.ndarray object, required
            It is usually a 1d vector with a shape of (T,).
        fps: int or float, optional
            Frame per second is a crucial recording parameter for your data. But it is 
            not very important to provide a exact fps value here. We only use this fps 
            parameter to correct the frame_interval.
        time_unit: str, optional, only ['ms', 's'] are valid values.
            The unit of behav_time. 'ms' represents millisecond while 's' represents second.

        Note
        ----
        This function is used to calculate the rate map of behavior. For this purpose, 
          the time stamp of related behavior data is required.
        
        '''
        if time_unit == 'ms':
            frame_interval = np.append(np.ediff1d(behav_time), 1000/fps)
        elif time_unit == 's':
            frame_interval = np.append(np.ediff1d(behav_time), 1/fps)
        else:
            print("Only 'ms' and 's' are valid value for parameter time_unit.")
            assert False
        
        _nbins = self.xbin * self.ybin
        _coords_range = [0, _nbins +0.0001 ]

        self.occu_time = scipy.stats.binned_statistic(self.behav_nodes,
                                                      frame_interval,
                                                      bins=_nbins,
                                                      statistic="sum",
                                                      range=_coords_range)