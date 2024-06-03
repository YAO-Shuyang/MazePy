import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from mylib.maze_utils3 import DrawMazeProfile, Clear_Axes

def RateMapAxes(
    ax: Axes,
    content: np.ndarray,
    maze_type: int,
    title: str="",
    is_plot_maze_walls: bool = True,
    is_inverty: bool=False,
    is_colorbar: bool=True,
    is_display_max_only: bool=True,
    colorbar_title: str = "Event Rate / Hz",
    nx: int = 48,
    imshow_args={'cmap': 'jet'},
    maze_args={'linewidth':1, 'color': 'white'},
) -> Axes:
    """
    RateMapAxes: to plot a rate map on a given axes.

    Parameters
    ----------
    ax : Axes
        The given axes.
    content : np.ndarray
        The content to be plotted onto the axes
    maze_type : int
        The maze type to provide a representation of maze's walls on the axes.
    title : str, optional
        The title of this figure, by default ""
    is_inverty : bool, optional
        Whether invert y axis, by default False
    is_plot_maze_walls: bool, optional
        Whether to plot the walls of a given sort of maze onto the axes or not, by default True
    is_colorbar : bool, optional
        Whether plotting a colorbar, by default True
    is_display_max_only : bool, optional
        Whether only display 0 and max value on the colorbar's ticks, by default True
    colorbar_title : str, optional
        The title of the colorbar, by default "Event Rate / Hz"
    nx : int, optional
        The size of the graph, by default 48

    Returns
    -------
    Axes
        The modified axes.
    """
    assert content.shape[0] == nx**2

    ax = Clear_Axes(ax)

    ax.set_aspect("equal")
    ax.set_title(title)
    
    if is_inverty:
        ax.invert_yaxis()
    
    im = ax.imshow(np.reshape(content, [nx, nx]), **imshow_args)
    ymax = np.nanmax(content)

    if is_plot_maze_walls:
        DrawMazeProfile(axes=ax, nx=nx, maze_type=maze_type, **maze_args)

    cbar = None
    if is_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        if is_display_max_only:
            cbar.set_ticks([0, ymax])
        cbar.set_label(colorbar_title)

    return ax, im, cbar