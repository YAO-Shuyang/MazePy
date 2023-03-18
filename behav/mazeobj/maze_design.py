import pyglet
import copy as cp

from mazepy.behav.mazeobj.windows import MainWindow
from mazepy.behav.graph import OpenFieldGraph

class DIYGraph(OpenFieldGraph):
    '''
    A do-it-yourself graph via GUI
    '''
    def __init__(self, xbin: int, ybin: int, **kwargs) -> None:
        super().__init__(xbin, ybin)
        self._setup_gui(**kwargs)

    def _setup_gui(self, **kwargs):
        MAIN = MainWindow(self.xbin, self.ybin, Graph = self.Graph, occu_map = self.occu_map, **kwargs)
        pyglet.clock.schedule_interval(MAIN.update, 1 / 60)
        pyglet.app.run()

        self.Graph = cp.deepcopy(MAIN.Graph)
        self.occu_map = cp.deepcopy(MAIN.occu_map)