'''
Date: March 13rd, 2023
Author: Shuyang Yao

Purpose: 
It could be used to divided recording parameters into bins. The parameters can be purely the location of particular animal, 
or a combination of different variables like time and location.
'''

try:
    import numpy as np
except:
    assert False


class Object():
    def __init__(self) -> None:
        pass

class BinVariable(Object):
    
    def __init__(self,
                 Mat: np.ndarray,
                 xmax: float|np.float64,
                 ymax: float|np.float64,
                 xbin: int = 1,
                 ybin: int = 1):
        '''
        Parameter
        ---------
        Mat: numpy.ndarray object, required
            It usually has a shape of (n, T), where n represent the dimension (default: 2) and 
            T represents the total number of frames that recorded.
        '''
        # We only accept dimension = 2 to divide variables into different bins.
        self.Mat = Mat
        self.xmax = xmax
        self.ymax = ymax
        self.xbin = xbin
        self.ybin = ybin

        
