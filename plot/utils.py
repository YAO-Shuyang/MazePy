"""Provide helpful function to assist users to plot beautiful figures."""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes

def clear_spines(ax: Axes, set_invisible_spines: list|str|None = None) -> Axes:
    """Function to set specific spines invisible.
    
    Parameter
    ---------
        ax: required, Axes object
            Contains the spines to be set invisible.
        set_invisible_spines: optional, default = None
            It can be a list of keys of spines that you want to set invisible, e.g., 
              ['top', 'right'] if you only want to set the top and right spine invisible.
              It can be str objects like 'all', 'All', 'a' and 'A' to express that you 
              want to set all of the four spines invisible.
    """
    
    if set_invisible_spines in ['all', 'All', 'a', 'A']:
        set_invisible_spines = ['top', 'bottom', 'right', 'left']
        
    elif set_invisible_spines is None:
        return ax
    
    for s in set_invisible_spines:
        assert s in ['top', 'bottom', 'right', 'left']
        ax.spines[s].set_visible(False)
        
    return ax

def save_figure(save_loc: str, formats: list = ['svg', 'png'], dpi: int = 600) -> None:
    """save_figure

    A function to save the figures.

    Parameters
    ----------
    save_loc : str
        The directory to save the figure.
    formats : list, optional
        Formats that you want to save your figures, by default ['svg', 'png']
    dpi : int, optional
        dots per inch that defines the quality of your figure, by default 600
    """
    for f in formats:
        plt.savefig(save_loc+'.'+f, dpi=dpi)
    