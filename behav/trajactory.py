"""_summary_

Provide an algorithm to rectify the trajactory of a animal from 
  coordinates of the points on mice labeled by deeplabcut to avoid
  potential cross-wall events. Actually, the cross-wall events are 
  quite common and might be .
"""

import numpy as np
import pickle
import warnings
import cv2
from mazepy.behav.graph import Graph
from mazepy.behav.affine import get_meanframe, PolygonDrawer

def mass(pts: np.ndarray):
    """
    Parameter
    ---------
        pts: required, shape (n, 2),
            Where n defines how many coordinates of the points that 
              have been labeled on the animal by deeplabcut.

    Note
    ----
        This function is written to calculate the coordinate of the 
          mass point of all the given point.
    """
    assert pts.shape[1] == 2

    return np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])



class Trajactory2D(object):
    def __init__(self, coordinate: np.ndarray) -> None:
        """
        Parameter
        ---------
            coordinate: required, numpy.ndarray, shape (T, 2)
                The coordinate of a animal recorded at each frame. 
        """
        assert coordinate.shape[1] == 2
        self._raw_coord = coordinate
        self._is_affine = False
        
        
    @property
    def raw_coord(self):
        """unprocessed trajactory"""
        return self._raw_coord
    
    def _process_overstepping_point(pts: np.ndarray, x_max: float, y_max: float, 
                                    x_min: float = 0, y_min: float = 0) -> np.ndarray:
        """
        Parameter
        ---------
            pts: required, numpy.ndarray, shape (T, 2)
                The trajactory points.
            
            x_max: required, float
            y_max: required, float
            x_min: optional, default = 0
            y_min: optional, default = 0
                These four floats define the range of trajactory. 
                  Points overstepping the range should be processed
                  either by deleting or by resetting. The latter one
                  is chosen in this function.
        """
        pts[np.where(pts[:, 0] < x_min)[0], 0] = x_min
        pts[np.where(pts[:, 1] < y_min)[0], 1] = y_min
        pts[np.where(pts[:, 0] >= x_max - 0.00001)[0], 0] = x_max - 0.00001 # Avoid overflowing
        pts[np.where(pts[:, 1] >= y_max - 0.00001)[0], 1] = y_max - 0.00001
        return pts
    
    def affine_transform(self, background: str, max_height: float, max_width: float, dtype: str = 'video'):
        """
        Parameter
        ---------
            background: str, required
                The directory of a figure/photo or video.
                You need to provide a background figure(photo) or video of 
                  where you recorded the trajactory of an animal. This would
                  offer you a reference to determine the edge and corner of 
                  the environment, which is required to perform affine trans-
                  formation.
                  
            max_height: required, float
            max_width: required, float
                The trajactory would be elongated or foreshortened linearly 
                  and independently in two dimensions (width and height). 
                  These two parameters define the max heigth and max width 
                  of the trajactory respectively, which function in this 
                  process.
                
            dtype: str, optional, default: 'video'
                The type of background you choose to input. It should match the
                  parameter 'background' or it will raise an error.
                Only 'video' and 'figure' are valid values.
        """
        self._is_affine = True
        
        if dtype == 'video':
            mean_frame = get_meanframe(background)
            equ_meanframe = cv2.equalizeHist(np.uint8(mean_frame))
        elif dtype == 'figure':
            img_color = cv2.imread(background)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            equ_meanframe = cv2.equalizeHist(np.uint8(img_gray))
        else:
            raise ValueError(f"Valid values include 'videl' and 'figure', but not {dtype}.")   
        
        # Affine transformation GUI
        pd = PolygonDrawer(equ_meanframe, self.raw_coord, max_height = max_height, max_width = max_width)
        self._warped_image, warped_positions, M  = pd.run()
        self.max_height, self.max_width = max_height, max_width
        
        # Restrict trajactory in the given range
        self._processed_coord = self._process_overstepping_point(warped_positions, max_width, max_height)
        
    
    @property
    def processed_coord(self):
        if self._is_affine:
            return self._processed_coord
        else:
            warnings.warn("The trajactory of the mouse has not been processed with affine transformation yet. Raw trajactory would be returned.")
            return self._raw_coord
        
    def plot_affine_transformedfig(self, save_loc: str):
        """
        Parameter:
            save_loc: required, str
                The directory of where you want to save the figure after affine transformation."""
        if self._is_affine:
            cv2.imwrite(save_loc, self._warped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            warnings.warn("The trajactory of the mouse has not been processed with affine transformation yet.")



class MultiTrajactory2D(object):
    def __init__(self, coordinate: np.ndarray) -> None:
        """
        Parameter
        ---------
            coordinate: numpy.ndarray, shape(T, n, 2)
                The coordinates of the points that have been labeled on 
                  an animal recorded at each frame by deeplabcut.
        """
        assert coordinate.shape[2] == 2
        
        self._n = coordinate.shape[1]
        self._raw_coord = coordinate
        
    @property
    def n(self):
        return self._n
    
    @property
    def raw_coord(self):
        """unprocessed trajactory"""
        return self._raw_coord
    
    def affine_transform(self, background: str, max_height: float, max_width: float, dtype: str = 'video'):
        """
        Parameter
        ---------
            background: str, required
                The directory of a figure/photo or video.
                You need to provide a background figure(photo) or video of 
                  where you recorded the trajactory of an animal. This would
                  offer you a reference to determine the edge and corner of 
                  the environment, which is required to perform affine trans-
                  formation.
                  
            max_height: required, float
            max_width: required, float
                The trajactory would be elongated or foreshortened linearly 
                  and independently in two dimensions (width and height). 
                  These two parameters define the max heigth and max width 
                  of the trajactory respectively, which function in this 
                  process.
                
            dtype: str, optional, default: 'video'
                The type of background you choose to input. It should match the
                  parameter 'background' or it will raise an error.
                Only 'video' and 'figure' are valid values.
        """
        if dtype == 'video':
            mean_frame = get_meanframe(background)
            equ_meanframe = cv2.equalizeHist(np.uint8(mean_frame))
        elif dtype == 'figure':
            img_color = cv2.imread(background)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            equ_meanframe = cv2.equalizeHist(np.uint8(img_gray))
        else:
            raise ValueError(f"Valid values include 'videl' and 'figure', but not {dtype}.")    
        
if __name__ == '__main__':
    with open(r'E:\Anaconda\envs\maze\Lib\site-packages\mazepy\tests\trace_behav.pkl', 'rb') as handle:
        trace = pickle.load(handle)
        
    import copy as cp
    import matplotlib.pyplot as plt
    
    ori_positions = cp.deepcopy(trace['ori_positions'])
    pos = Trajactory2D(ori_positions)
    print(ori_positions.shape)
    pos.affine_transform(r'E:\Anaconda\envs\maze\Lib\site-packages\mazepy\tests\0.avi', max_height=960, max_width=960, dtype = 'video')
    
    
    