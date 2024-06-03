import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from mylib.maze_utils3 import DrawMazeProfile, Clear_Axes, ColorBarsTicks
import sklearn.preprocessing
from mylib.maze_graph import correct_paths, incorrect_paths, xorders

def sort_x(contents: np.ndarray, x_order: np.ndarray):
    return contents[:, x_order]

def get_y_order(contents: np.array):
    x_max = np.nanargmax(contents, axis=1)
    assert contents.shape[0] == x_max.shape[0]

    y_order = np.array([], np.int64)
    
    for i in range(contents.shape[1]):
        y_order = np.concatenate([y_order, np.where(x_max == i)[0]])

    return y_order

def sort_y(contents: np.ndarray):
    y_order = get_y_order(contents=contents)
    return contents[y_order, :]

def normalization(contents: np.ndarray):
    return sklearn.preprocessing.minmax_scale(contents, feature_range=(0, 1), axis=1, copy=True)
    

def PeakCurveAxes(
    ax: Axes,
    contents: np.ndarray,
    maze_type: int,
    title: str = '',
    path_type: str = 'cp',
    y_order: np.ndarray | None = None,
    imshow_args: dict = {},
    is_inverty: bool = True,
    is_invertx: bool = False,
    is_sortx: bool = True
) -> Axes:
    """
    PeakCurveAxes: the trace of the spatial distribution of place cell's main field

    Parameters
    ----------
    ax : Axes
        The given axes.
    content : np.ndarray
        The content to be plotted onto the axes
    maze_type : int
        The maze type to provide a representation of maze's walls on the axes.
    path_type : str, optional
        The maze track selected to plot this figure, by default 'cp'
        You can select 'cp' (means correct path), 'ip' (means incorrect path), 
          and 'all' (means both). Note that if you choose 'all', it would automatedly 
          arrange correct path at first and then the incorrect path.
    y_order : np.ndarray | None, optional
        To arrange the y axis order by a given order vector, by default None
    imshow_args : dict, optional
        Additional parameters to adjust the effect of imshow function, by default {}
    is_inverty : bool, optional
        Whether invert y axis, by default True
    is_invertx : bool, optional
        Whether invert x axis, by default False

    Returns
    -------
    Axes
        This modified axes object.
    """
    assert path_type in ['cp', 'ip', 'all']

    if path_type == 'cp':
        x_order = correct_paths[int(maze_type)]-1
        x_ticks = [-0.5, len(x_order)/2, len(x_order)-0.5]
        x_label = ['start', 'correct track', 'end']
        divide_line = None
    elif path_type == 'ip':
        x_order == incorrect_paths[int(maze_type)]-1
        x_ticks = [-0.5, len(x_order)/2, len(x_order)-0.5]
        x_label = ['start', 'correct track', 'end']
        divide_line = None
    else:
        x_order = xorders[int(maze_type)]-1
        x_ticks = [-0.5, len(correct_paths[int(maze_type)])/2, len(correct_paths[int(maze_type)]) + len(incorrect_paths[int(maze_type)])/2, len(x_order)-0.5]
        x_label = ['start', 'correct track', 'incorrect_track', 'end']
        divide_line = len(correct_paths[int(maze_type)]) - 0.5

    if is_sortx:
        contents = sort_x(contents, x_order)
        
    contents = normalization(contents)

    if y_order is None:
        contents = sort_y(contents)
    else:
        contents = contents[y_order, :]

    n_neuron = contents.shape[0]

    ax.imshow(contents)
    ax.set_title(title)
    ax.set_yticks([0, n_neuron-1], [1, n_neuron])
    ax.set_xticks(ticks = x_ticks, labels = x_label)

    if divide_line is not None:
        ax.axvline(divide_line, color = 'orange')

    if is_inverty:
        ax.invert_yaxis()

    if is_invertx:
        ax.invert_xaxis()

    ax.set_aspect("auto")

    return ax
    


    