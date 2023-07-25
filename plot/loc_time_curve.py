import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from mylib.maze_utils3 import Clear_Axes
from mylib.maze_graph import NRG, correct_paths

def LocTimeCurveAxes(
    ax: Axes,
    behav_time: np.ndarray,
    spikes: np.ndarray,
    spike_time: np.ndarray,
    maze_type: int,
    behav_nodes: np.ndarray | None = None,
    given_x: np.ndarray | None = None,
    title: str = "",
    title_color: str = "black",
    is_invertx: bool = False
) -> tuple[Axes, list, list]:

    ax = Clear_Axes(axes=ax, close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
    ax.set_aspect("auto")

    if given_x is None:
        assert behav_nodes is not None
        linearized_x = np.zeros_like(behav_nodes, np.float64)
        graph = NRG[int(maze_type)]

        for i in range(behav_nodes.shape[0]):
            linearized_x[i] = graph[int(behav_nodes[i])]
    
        linearized_x = linearized_x + np.random.rand(behav_nodes.shape[0]) - 0.5
    else:
        linearized_x = given_x
    
    spike_burst_time = spike_time[np.where(spikes == 1)[0]]
    spike_loc_id = np.zeros_like(spike_burst_time, dtype=np.int64)

    for i, t in enumerate(spike_burst_time):
        try:
            spike_loc_id[i] = np.where(behav_time>=t)[0][0]
        except:
            spike_loc_id[i] = np.where(behav_time<t)[0][-1]

    x_spikes = linearized_x[spike_loc_id]
    t_spikes = behav_time[spike_loc_id]

    t_max = int(np.nanmax(behav_time)/1000)

    a = ax.plot(linearized_x, behav_time/1000, 'o', markeredgewidth = 0, markersize = 1, color = 'black')
    b = ax.plot(x_spikes, t_spikes/1000, '|', color='red', markeredgewidth = 1, markersize = 4)

    ax.set_title(title, color=title_color)
    ax.set_xticks([1, len(correct_paths[int(maze_type)])/2, len(correct_paths[int(maze_type)])], labels = ['start', 'correct track', 'end'])
    ax.set_yticks([0, t_max])
    ax.set_ylabel("Time / s")

    if is_invertx:
        ax.invert_xaxis()

    return ax, a, b