import numpy as np
import copy as cp
from tqdm import tqdm
from mylib.maze_utils3 import maze_graphs


# generate all subfield. ============================================================================
# place field analysis, return a dict contatins all field. If you want to know the field number of a certain cell, you only need to get it by use 
# len(trace['place_field_all'][n].keys())
def GeneratePlaceField(maze_type: int, nx: int = 48, smooth_map: np.ndarray = None) -> dict:
    """
    GeneratePlaceField: to generate place fields of a place cell.

    Parameters
    ----------
    maze_type : int
        _description_
    nx : int, optional
        _description_, by default 48
    smooth_map : np.ndarray, optional
        _description_, by default None

    Returns
    -------
    dict
        _description_
    """
    # rate_map should be one without NAN value. Use function clear_NAN(rate_map_all) to process first.
    MAX = max(smooth_map)
    field_set = np.where(smooth_map >= 0.5*MAX)[0]+1
    search_set = []
    All_field = {}

    while len(np.setdiff1d(field_set, search_set))!=0:
        diff = np.setdiff1d(field_set,search_set)
        point = diff[0]
        subfield = field(rate_map = smooth_map, point = point, maze_type = maze_type, nx = nx, MAX = MAX)
        peak_loc = subfield[0]
        peak = smooth_map[peak_loc-1]
        # find peak idx as keys of place_field_all dict objects.
        for k in subfield:
            if smooth_map[k-1] > peak:
                peak = smooth_map[k-1]
                peak_loc = k
        All_field[peak_loc] = subfield
        search_set = sum([search_set, subfield],[])
    
    return All_field
               
def field(rate_map = None, point = 1, maze_type = 1, nx = 48,MAX = 0):
    if (maze_type, nx) in maze_graphs.keys():
        graph = maze_graphs[(maze_type, nx)]
    else:
        assert False
            
    MaxStep = 300
    step = 0
    Area = [point]
    StepExpand = {0: [point]}
    while step <= MaxStep:
        StepExpand[step+1] = []
        for k in StepExpand[step]:
            surr = graph[k]
            for j in surr:
                if rate_map[j-1] >= 0.5*MAX and j not in Area:
                    StepExpand[step+1].append(j)
                    Area.append(j)
        
        # Generate field successfully! 
        if len(StepExpand[step+1]) == 0:
            break
            
        step += 1
    return Area

# get all cell's place field
def place_field(n_neuron = None, smooth_map_all = None, maze_type = 1):
    place_field_all = []
    smooth_map_all = cp.deepcopy(smooth_map_all)
    for k in tqdm(range(n_neuron)):
        place_field = GeneratePlaceField(smooth_map = smooth_map_all[k], maze_type = maze_type, nx = 48)
        place_field_all.append(place_field)
    print("    Place field has been generated successfully.")
    return place_field_all