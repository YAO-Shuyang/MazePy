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
    print()


class BinVariable(np.ndarray):
    