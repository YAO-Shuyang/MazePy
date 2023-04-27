'''
Some environments have relative complex internal structure, while others are not.

We provide some tools for users to define the internal structure. It contains a
  GUI to help you manually select where to place your internal structure (e.g., 
  Walls or objects)
'''

from mazepy.plot.mazeprofile import MazeProfile
from mazepy.behav.mazeobj.maze_design import DIYGraph

if __name__ == '__main__':
    G = DIYGraph(xbin=12, ybin=12, width = 5)
    mazeprof = MazeProfile(xbin=12, ybin=12, Graph=G.Graph, occu_map=G.occu_map, color = 'yellow', linewidth = 2)