import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from mylib.maze_utils3 import DrawMazeProfile, Clear_Axes

def insert_temporary_nan(trajectory: np.ndarray, behav_time: np.ndarray, thre: float = 400):
    dt = np.ediff1d(behav_time)
    dx = np.ediff1d(trajectory[:, 0])
    dy = np.ediff1d(trajectory[:, 1])
    dl = np.sqrt(dx**2 + dy**2)
    behav_time = behav_time.astype(np.float64)

    idx = np.where(dt > thre)[0]
    return np.insert(trajectory, idx+1, [np.nan, np.nan], axis=0), np.insert(behav_time, idx+1, np.nan)


def TraceMapAxes(
    ax: Axes,
    trajectory: np.ndarray,
    behav_time: np.ndarray,
    spikes: np.ndarray,
    spike_time: np.ndarray,
    maze_type: int = 0,
    is_plot_maze_walls: bool = True,
    title: str="",
    is_inverty: bool=False,
    maze_args={'linewidth':1, 'color': 'black'},
) -> tuple:
    """
    TraceMapAxes: plot the trajectory of the animal and project the detected events onto the trajectory.

    Parameters
    ----------
    ax : Axes
        The axes object to bear the figure.
    trajectory : np.ndarray
        The x, y value recorded by top-view cameras.
    behav_time : np.ndarray
        The behavioral timestamp.
    spikes : np.ndarray
        The recorded event trains.
    spike_time : np.ndarray
        The timestamp of the event trains.
    maze_type : int
        The maze type to provide a representation of maze's walls 
        on the axes.
    is_plot_maze_walls: bool, optional
        Whether to plot the walls of a given sort of maze onto the 
        axes or not, by default True
    title : str, optional
        The title of this figure, by default ""
    is_inverty : bool, optional
        Whether invert y axis, by default False
    maze_args : dict, optional
        The additional arguments to set properties of the maze's 
        walls which is to be plotted, by default:
                 {'linewidth':1, 'color': 'black'}

    Returns
    -------
    tuple
        tuple(axes, list, list)
    """
    
    ax = Clear_Axes(axes=ax)
    ax.set_aspect("equal")

    trajectory[:, 0] = trajectory[:, 0]/20 - 0.5
    trajectory[:, 1] = trajectory[:, 1]/20 - 0.5
    trajectory, behav_time = insert_temporary_nan(trajectory, behav_time)

    a = ax.plot(trajectory[:, 0], trajectory[:, 1], 'gray')
    
    spike_burst_time = spike_time[np.where(spikes == 1)[0]]
    spike_loc_id = np.zeros_like(spike_burst_time, dtype=np.int64)

    for i, t in enumerate(spike_burst_time):
        try:
            spike_loc_id[i] = np.where(behav_time>=t)[0][0]
        except:
            spike_loc_id[i] = np.where(behav_time<t)[0][-1]

    x_spike, y_spike = trajectory[spike_loc_id, 0], trajectory[spike_loc_id, 1]

    b = ax.plot(x_spike, y_spike, 'o', color = 'black', markeredgewidth=0, markersize=4)

    if is_plot_maze_walls:
        ax = DrawMazeProfile(axes=ax, maze_type=maze_type, nx = 48, **maze_args)

    if is_inverty:
        ax.invert_yaxis()

    ax.set_title(title)

    return ax, a, b
