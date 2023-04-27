<<<<<<< HEAD
"""
This file provide the graph of open field. This kind of graph is 
  called basic graph. Additional modifications will be added on 
  this basic graph via manually setting up on GUI.

Graph is the fundamental object to depict the property of a certain 
  environment. It illustrates the connections between each two squared 
  bins, and modifications on the graph are to modify these connections.

Here we offer the basic graph based on open field with a shape of (xbin, ybin)

Border:
        North
          ↑
    West ← → East
          ↓
        South

Note that bin ID (as we have defined in /behav/trasloc.py) belongs to {1, 2, ..., xbin*ybin}

class Graph defines the basic properties and algorithms on the environment, such the distance
  between any two points in the environment. It cannot simply be determined by cartesian 
  distance, sometimes. So we provide a algorithm that can relative fast calculated the 
  'maze distance'. This algorithm is what we think the fastest one.
"""


import numpy as np
import copy as cp
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.axes._axes import Axes

from mazepy.behav.grid import GridBasic
from mazepy.behav.transloc import idx_to_loc, pvl_to_idx
from mazepy.behav.transloc import isNorthBorder, isSouthBorder, isEastBorder, isWestBorder
from mazepy.behav.element import Point, Edge, Bin

def WallMatrix(Graph: dict, xbin: int, ybin: int) -> None:
    """
    Return
    ------
    numpy.ndarray Matrix * 2
    Wall matrix will return 2 matrix: vertical_walls represents verticle walls which has a 
      shape of (xbin+1, ybin) and horizont_walls with a shape of (xbin, ybin+1).
      1 -> there's a wall; 0 -> there's no wall.
    """
    vertical_walls = np.ones((ybin, xbin+1), dtype = np.int64)
    horizont_walls = np.ones((ybin+1, xbin), dtype = np.int64)

    for i in range(1, xbin*ybin):
        x, y = idx_to_loc(i, xbin=xbin, ybin=ybin)

        surr = Graph[i]
        for s in surr:
            if s == i + 1:
                vertical_walls[y, x+1] = 0
            elif s == i - 1:
                vertical_walls[y, x] = 0
            elif s == i + xbin:
                horizont_walls[y+1, x] = 0
            elif s == i - xbin:
                horizont_walls[y, x] = 0
            else:
                raise ValueError(f"Bin {s} is not connected with Bin {i}.")
            
    return vertical_walls, horizont_walls


class OpenFieldGraph(GridBasic):
    '''
    Generate the graph of open field. The basic information of the grids should be given.
    '''
    def __init__(self, xbin: int, ybin: int) -> None:
        super().__init__(xbin = xbin, ybin = ybin)
        self.Graph = {}
        self._generate_graph()
        self.occu_map = np.zeros(xbin*ybin, dtype=np.float64)
    
    def _generate_graph(self) -> None:
        '''
        Note
        ----
        To generate the graph via enumeration of each bin.
        '''
        for ID in range(1,self.xbin*self.ybin + 1):
            self.Graph[ID] = []
            if isNorthBorder(ID, xbin=self.xbin, ybin=self.ybin) == False:
                self.Graph[ID].append(ID + self.xbin)
            if isSouthBorder(ID, xbin=self.xbin) == False:
                self.Graph[ID].append(ID - self.xbin)
            if isEastBorder(ID, xbin=self.xbin) == False:
                self.Graph[ID].append(ID + 1)
            if isWestBorder(ID, xbin=self.xbin) == False:
                self.Graph[ID].append(ID - 1)

            self.Graph[ID] = np.array(self.Graph[ID], dtype = np.int64)




class Graph(GridBasic):
    def __init__(self, xbin: int, ybin: int, Graph: dict) -> None:
        super().__init__(xbin, ybin)
        if len(Graph.keys()) != xbin*ybin:
            raise ValueError(f"""The Graph does not match the environment. 
            Graph.keys() should have a length of {xbin*ybin}, but {len(Graph.keys())} however.""")
        
        self.Graph = Graph
        self.OFGraph = OpenFieldGraph(self.xbin, self.ybin)
        self._WallMatrix()
        self._init_corner_graph()

    def _WallMatrix(self):
        self._vertical_walls, self._horizont_walls = WallMatrix(self.Graph, self.xbin, self.ybin)
        return self._vertical_walls, self._horizont_walls
    
    @property
    def vertical_wall(self):
        return self._vertical_walls
    
    @property
    def horizont_wall(self):
        return self._horizont_walls
    
    def _cartesian(self, p1: tuple, p2: tuple):
        return np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

    def _is_on_point(self, p: tuple):
        if p[0] == int(p[0]) and p[1] == int(p[1]):
            return True
        else:
            return False
        
    def _is_on_edge(self, p: tuple):
        if (p[0] == int(p[0]) and p[1] != int(p[1])) or (p[0] != int(p[0]) and p[1] == int(p[1])):
            return True
        else:
            return False
        
    def _vertical_cross_wall(self, p1: tuple, p2:tuple, dirc: str, step_length):
        """
        x1 = x2 = int(x1) = int(x2)
        """
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

        if self._cartesian(p1, p2) >= step_length:
            x_set = np.linspace(p2[0], p1[0], int(self._cartesian(p1, p2) / step_length) + 2)
            y_set = np.linspace(p2[1], p1[1], int(self._cartesian(p1, p2) / step_length) + 2)
            for i in range(x_set.shape[0]-1):
                if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), step_length=step_length):
                    return True

            return False

    def iscross_wall(self, p1: tuple, p2: tuple, step_length: float = 1, left_or_right = 'l', up_and_down = 'u') -> bool:        
        if p1[0] >= self.xbin or p2[0] >= self.xbin or p1[0] < 0 or p2[0] < 0:
            raise ValueError(f"{p1[0]} is not a valid value in the maze. It should belong to [0, {self.xbin}).")
        
        if p1[1] >= self.ybin or p2[1] >= self.ybin or p1[1] < 0 or p2[1] < 0:
            raise ValueError(f"{p1[1]} is not a valid value in the maze. It should belong to [0, {self.ybin}).")
        
        if p1 == p2:
            return False


        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

        #else:  # (x1 - x2)*(y1 - y2) != 0 or (x1 == x2 and x1 != len(x1)) or (y1 == y2 and y1 != len(y1))

        if self._cartesian(p1, p2) >= step_length:
            
            x_set = np.linspace(p2[0], p1[0], int(self._cartesian(p1, p2) / step_length) + 2)
            y_set = np.linspace(p2[1], p1[1], int(self._cartesian(p1, p2) / step_length) + 2)

            self._l, self._r = False, False
            self._u, self._d = False, False
            for i in range(x_set.shape[0]-1):
                if x1 == x2 and x1 == int(x1):
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), left_or_right = 'l'):
                        self._l = True
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), left_or_right = 'r'):
                        self._r = True
                    
                    if self._l and self._r:
                        return True

                    E = Edge(xbin=self.xbin, ybin=self.ybin, x=x1, y=np.min([int(y_set[i+1]), int(y_set[i])]), dirc='v')
                    E.place_wall(self.Graph)
                    if E.Wall == False:
                        self._r = self._l = False

                elif y1 == y2 and y1 == int(y1):
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), up_and_down = 'u'):
                        self._u = True
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), up_and_down = 'd'):
                        self._d = True

                    if self._u and self._d:
                        return True

                    E = Edge(xbin=self.xbin, ybin=self.ybin, x=np.min([int(x_set[i+1]), int(x_set[i])]), y=y1, dirc='h')
                    E.place_wall(self.Graph)
                    if E.Wall == False:
                        self._u = self._d = False
                    
                else:
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), step_length=step_length):
                        return True

            return False

        else:
            id1 = pvl_to_idx(x1, y1, xbin=self.xbin, ybin=self.ybin, xmax=self.xbin, ymax=self.ybin)
            id2 = pvl_to_idx(x2, y2, xbin=self.xbin, ybin=self.ybin, xmax=self.xbin, ymax=self.ybin)

            if x1 == x2 and x1 == int(x1):
                if self._is_on_point(p1) and self._is_on_point(p2):
                    P1 = Point(self.xbin, self.ybin, x1, y1)
                    P2 = Point(self.xbin, self.ybin, x2, y2)
                    E = Edge(self.xbin, self.ybin, x1, np.min([int(y1), int(y2)]), 'v')
                    P1.place_wall(self.Graph)
                    P2.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P1.EWall == False and P2.EWall == False and left_or_right == 'r':
                            return False
                        elif P1.WWall == False and P2.WWall == False and left_or_right == 'l':
                            return False
                        else:
                            return True

                elif self._is_on_point(p1) == True and self._is_on_point(p2) == False:
                    P1 = Point(self.xbin, self.ybin, x1, y1)
                    E = Edge(self.xbin, self.ybin, x1, np.min([int(y1), int(y2)]), 'v')
                    P1.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P1.EWall == False and left_or_right == 'r':
                            return False
                        elif P1.WWall == False and left_or_right == 'l':
                            return False
                        else:
                            return True
                
                elif self._is_on_point(p1) == False and self._is_on_point(p2) == True:
                    P2 = Point(self.xbin, self.ybin, x2, y2)
                    E = Edge(self.xbin, self.ybin, x1, np.min([int(y1), int(y2)]), 'v')
                    P2.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P2.EWall == False and left_or_right == 'r':
                            return False
                        elif P2.WWall == False and left_or_right == 'l':
                            return False
                        else:
                            return True         

                else: # Both of p1 and p2 are on edges instead of points.
                    if id1 == id2:
                        return False
                    else:                
                        P = Point(self.xbin, self.ybin, x1, np.max([int(y1), int(y2)]))
                        P.place_wall(self.Graph)

                        if P.EWall == False and left_or_right == 'r':
                            return False
                        if P.WWall == False and left_or_right == 'l':
                            return False
                        else:
                            return True
            
            elif y1 == y2 and y1 == int(y1):

                if self._is_on_point(p1) and self._is_on_point(p2):
                    P1 = Point(self.xbin, self.ybin, x1, y1)
                    P2 = Point(self.xbin, self.ybin, x2, y2)
                    E = Edge(self.xbin, self.ybin, np.min([int(x1), int(x2)]), y1, 'h')
                    P1.place_wall(self.Graph)
                    P2.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P1.NWall == False and P2.NWall == False and up_and_down == 'u':
                            return False
                        elif P1.SWall == False and P2.SWall == False and up_and_down == 'd':
                            return False
                        else:
                            return True

                elif self._is_on_point(p1) == True and self._is_on_point(p2) == False:
                    P1 = Point(self.xbin, self.ybin, x1, y1)
                    E = Edge(self.xbin, self.ybin, np.min([int(x1), int(x2)]), y1, 'h')
                    P1.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P1.NWall == False and up_and_down == 'u':
                            return False
                        elif P1.SWall == False and up_and_down == 'd':
                            return False
                        else:
                            return True
                
                elif self._is_on_point(p1) == False and self._is_on_point(p2) == True:
                    P2 = Point(self.xbin, self.ybin, x2, y2)
                    E = Edge(self.xbin, self.ybin, np.min([int(x1), int(x2)]), y1, 'h')
                    P2.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P2.NWall == False and up_and_down == 'u':
                            return False
                        elif P2.SWall == False and up_and_down == 'd':
                            return False
                        else:
                            return True    

                else: # Both of p1 and p2 are on edges instead of points.
                    if id1 == id2:
                        return False
                    else:                
                        P = Point(self.xbin, self.ybin, np.max([int(x1), int(x2)]), y1)
                        P.place_wall(self.Graph)

                        if P.NWall == False and up_and_down == 'u':
                            return False
                        if P.SWall == False and up_and_down == 'd':
                            return False
                        else:
                            return True             
                
            else:
                if id1 == id2:
                    return False
                
                if self._is_on_point(p1):
                    P = Point(self.xbin, self.ybin, x1, y1)
                    return True ^ P.is_passable(self.Graph, x2, y2)

                if self._is_on_point(p2):
                    P = Point(self.xbin, self.ybin, x2, y2)
                    return True ^ P.is_passable(self.Graph, x1, y1)
                
                if id1 not in self.OFGraph.Graph[id2]:
                    P = Point(self.xbin, self.ybin, np.max([int(x2), int(x1)]), np.max([int(y2), int(y1)]))
                    P.place_wall(Graph=self.Graph)

                    k, b = (y2 - y1)/(x2 - x1), ((y1 - P.y)*(x2 - P.x) - (y2 - P.y)*(x1 - P.x))/(x2 - x1)

                    if k > 0:
                        if b > 0:
                            return P.NWall or P.WWall
                        elif b == 0:
                            return True ^ P.is_passable(self.Graph, x1, y1)
                        else: # b < 0
                            return P.SWall or P.EWall
                    
                    elif k < 0:
                        if b > 0:
                            return P.NWall or P.EWall
                        elif b == 0:
                            return True ^ P.is_passable(self.Graph, x1, y1)
                        else: # b < 0
                            return P.SWall or P.WWall
                
                    # if x1 = x2 or y1 = y2, the id1 must be listed in self.OFGraph[id2], 
                    # so this case should not be taken into concerns.

                else:
                    if id1 not in self.Graph[id2]:
                        return True
                    else:
                        return False
                return False
    
    def _init_corner_graph(self):
        """
        We assumed that the shortest distance between any two points (denote as A and B) in the 
          maze must have the following forms: A -> (some turns) -> B
        
        The turns that listed here are specifically refer to points (Class Point) which satisfy
          following criteria:
          1. It should connect with 4 edges. (if only connect with 3 edges, it is on the brink of 
             the maze. If only connect with 2 edges, the point is at the corner of the maze.)
          2. It could at most connect with 2 walls.
          3. If it connects with 1 wall, it is a 'turn'. If it connects with 2 walls, the two walls
             must be placed at a 90° angle.

        The aim of this function is to find all of the turns point and connect them to build a graph.
        """
        Ps = self._find_all_corner_points()
        G = np.zeros([Ps.shape[0], Ps.shape[0]])
        for i in range(Ps.shape[0]-1):
            for j in range(i+1, Ps.shape[0]):
                if self.iscross_wall(p1 = (self.Ps[i, 0], self.Ps[i, 1]), p2 = (self.Ps[j, 0], self.Ps[j, 1])) == False:
                        if self.Points[i].is_passable(self.Graph, Ps[j, 0], Ps[j, 1]) and self.Points[j].is_passable(self.Graph, Ps[i, 0], Ps[i, 1]):
                            G[i,j] = G[j,i] = self._cartesian((self.Ps[i, 0], self.Ps[i, 1]), (self.Ps[j, 0], self.Ps[j, 1]))

        self.ConnectMat = G
        self.CornerGraph = nx.Graph(G)
        print(self.CornerGraph)
    
    def _find_all_corner_points(self):
        self.Ps = []
        self.Points = [] 
        for i in range(self.xbin+1):
            for j in range(self.ybin+1):
                P = Point(self.xbin, self.ybin, i, j)
                if P.is_corner_point(self.Graph):
                    self.Ps.append([P.x, P.y])
                    self.Points.append(P)
        self.Ps = np.array(self.Ps)
        return self.Ps

    def shortest_distance(self, p1: tuple, p2: tuple, **kwargs):
        """
        Using dijkstra algorithm to find the shortest path length between two point.
        """
        if self.iscross_wall(p1, p2) == False:
            return self._cartesian(p1, p2)

        G = cp.deepcopy(self.CornerGraph)
        nodes = len(G.nodes)

        G.add_node(nodes)
        G.add_node(nodes+1)
        for i in range(nodes):
            if self.iscross_wall(p1, (self.Ps[i, 0], self.Ps[i, 1])) == False:
                if self.Points[i].is_passable(self.Graph, p1[0], p1[1]):
                    G.add_edge(i, nodes, weight = self._cartesian(p1, (self.Ps[i, 0], self.Ps[i, 1])))
            
            if self.iscross_wall(p2, (self.Ps[i, 0], self.Ps[i, 1])) == False:
                if self.Points[i].is_passable(self.Graph, p2[0], p2[1]):
                    G.add_edge(i, nodes+1, weight = self._cartesian(p2, (self.Ps[i, 0], self.Ps[i, 1])))

        return nx.dijkstra_path_length(G=G, source=nodes, target=nodes+1, **kwargs)
    
    def shortest_path(self, p1: tuple, p2: tuple, **kwargs):
        G = cp.deepcopy(self.CornerGraph)
        nodes = len(G.nodes)

        if self.iscross_wall(p1, p2) == False:
            return [nodes, nodes+1]

        G.add_node(nodes)
        G.add_node(nodes+1)
        for i in range(nodes):
            if self.iscross_wall(p1, (self.Ps[i, 0], self.Ps[i, 1])) == False:
                if self.Points[i].is_passable(self.Graph, p1[0], p1[1]):
                    G.add_edge(i, nodes, weight = self._cartesian(p1, (self.Ps[i, 0], self.Ps[i, 1])))
            
            if self.iscross_wall(p2, (self.Ps[i, 0], self.Ps[i, 1])) == False:
                if self.Points[i].is_passable(self.Graph, p2[0], p2[1]):
                    G.add_edge(i, nodes+1, weight = self._cartesian(p2, (self.Ps[i, 0], self.Ps[i, 1])))

        return nx.dijkstra_path(G=G, source=nodes, target=nodes+1, **kwargs)

    def plot_shortest_path(self, p1: tuple, p2:tuple, ax: Axes = None, dx: float = 0.5, dy: float = 0.5,
                           figsize: tuple = (6, 6), color = 'red', linewidth = 2, **kwargs) -> Axes:
        if ax is None:
            fig = plt.figure(figsize = figsize)
            ax = plt.axes()
        
        path = self.shortest_path(p1, p2)

        if len(path) > 2:

            ax.plot([p1[0] - dx, self.Ps[path[1], 0] - dx], [p1[1] - dy, self.Ps[path[1], 1] - dy], color = color, linewidth = 2, **kwargs)
            for i in range(1, len(path)-2):
                #ax.text(self.Ps[path[i], 0] - dx, self.Ps[path[i], 1] - dx, str(path[i]))
                ax.plot([self.Ps[path[i], 0] - dx, self.Ps[path[i+1], 0] - dx], [self.Ps[path[i], 1] - dy, self.Ps[path[i+1], 1] - dy], 
                         color = color, linewidth = linewidth, **kwargs)
            ax.plot([self.Ps[path[-2], 0] - dx, p2[0] - dx], [self.Ps[path[-2], 1] - dy, p2[1] - dy], color = color, linewidth = linewidth, **kwargs)
        else: # len(path) <= 2:
            #ax.text(self.Ps[path[-2], 0] - dx, self.Ps[path[-2], 1] - dx, str(path[-2]))
            ax.plot([p1[0] - dx, p2[0] - dx], [p1[1] - dy, p2[1] - dy], color = color, linewidth = linewidth, **kwargs)

        for i in range(self.Ps.shape[0]):
            if i in path:
                ax.text(self.Ps[i, 0] - dx, self.Ps[i, 1] - dy, str(i), color = 'black')
            else:
                ax.text(self.Ps[i, 0] - dx, self.Ps[i, 1] - dy, str(i), color = 'gray')
        
        return ax


=======
"""
This file provide the graph of open field. This kind of graph is 
  called basic graph. Additional modifications will be added on 
  this basic graph via manually setting up on GUI.

Graph is the fundamental object to depict the property of a certain 
  environment. It illustrates the connections between each two squared 
  bins, and modifications on the graph are to modify these connections.

Here we offer the basic graph based on open field with a shape of (xbin, ybin)

Border:
        North
          ↑
    West ← → East
          ↓
        South

Note that bin ID (as we have defined in /behav/trasloc.py) belongs to {1, 2, ..., xbin*ybin}

class Graph defines the basic properties and algorithms on the environment, such the distance
  between any two points in the environment. It cannot simply be determined by cartesian 
  distance, sometimes. So we provide a algorithm that can relative fast calculated the 
  'maze distance'. This algorithm is what we think the fastest one.
"""


import numpy as np
import copy as cp
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.axes._axes import Axes

from mazepy.behav.grid import GridBasic
from mazepy.behav.transloc import idx_to_loc, pvl_to_idx
from mazepy.behav.transloc import isNorthBorder, isSouthBorder, isEastBorder, isWestBorder
from mazepy.behav.element import Point, Edge, Bin

def WallMatrix(Graph: dict, xbin: int, ybin: int) -> None:
    """
    Return
    ------
    numpy.ndarray Matrix * 2
    Wall matrix will return 2 matrix: vertical_walls represents verticle walls which has a 
      shape of (xbin+1, ybin) and horizont_walls with a shape of (xbin, ybin+1).
      1 -> there's a wall; 0 -> there's no wall.
    """
    vertical_walls = np.ones((ybin, xbin+1), dtype = np.int64)
    horizont_walls = np.ones((ybin+1, xbin), dtype = np.int64)

    for i in range(1, xbin*ybin):
        x, y = idx_to_loc(i, xbin=xbin, ybin=ybin)

        surr = Graph[i]
        for s in surr:
            if s == i + 1:
                vertical_walls[y, x+1] = 0
            elif s == i - 1:
                vertical_walls[y, x] = 0
            elif s == i + xbin:
                horizont_walls[y+1, x] = 0
            elif s == i - xbin:
                horizont_walls[y, x] = 0
            else:
                raise ValueError(f"Bin {s} is not connected with Bin {i}.")
            
    return vertical_walls, horizont_walls


class OpenFieldGraph(GridBasic):
    '''
    Generate the graph of open field. The basic information of the grids should be given.
    '''
    def __init__(self, xbin: int, ybin: int) -> None:
        super().__init__(xbin = xbin, ybin = ybin)
        self.Graph = {}
        self._generate_graph()
        self.occu_map = np.zeros(xbin*ybin, dtype=np.float64)
    
    def _generate_graph(self) -> None:
        '''
        Note
        ----
        To generate the graph via enumeration of each bin.
        '''
        for ID in range(1,self.xbin*self.ybin + 1):
            self.Graph[ID] = []
            if isNorthBorder(ID, xbin=self.xbin, ybin=self.ybin) == False:
                self.Graph[ID].append(ID + self.xbin)
            if isSouthBorder(ID, xbin=self.xbin) == False:
                self.Graph[ID].append(ID - self.xbin)
            if isEastBorder(ID, xbin=self.xbin) == False:
                self.Graph[ID].append(ID + 1)
            if isWestBorder(ID, xbin=self.xbin) == False:
                self.Graph[ID].append(ID - 1)

            self.Graph[ID] = np.array(self.Graph[ID], dtype = np.int64)




class Graph(GridBasic):
    def __init__(self, xbin: int, ybin: int, Graph: dict) -> None:
        super().__init__(xbin, ybin)
        if len(Graph.keys()) != xbin*ybin:
            raise ValueError(f"""The Graph does not match the environment. 
            Graph.keys() should have a length of {xbin*ybin}, but {len(Graph.keys())} however.""")
        
        self.Graph = Graph
        self.OFGraph = OpenFieldGraph(self.xbin, self.ybin)
        self._WallMatrix()
        self._init_corner_graph()

    def _WallMatrix(self):
        self._vertical_walls, self._horizont_walls = WallMatrix(self.Graph, self.xbin, self.ybin)
        return self._vertical_walls, self._horizont_walls
    
    @property
    def vertical_wall(self):
        return self._vertical_walls
    
    @property
    def horizont_wall(self):
        return self._horizont_walls
    
    def _cartesian(self, p1: tuple, p2: tuple):
        return np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

    def _is_on_point(self, p: tuple):
        if p[0] == int(p[0]) and p[1] == int(p[1]):
            return True
        else:
            return False
        
    def _is_on_edge(self, p: tuple):
        if (p[0] == int(p[0]) and p[1] != int(p[1])) or (p[0] != int(p[0]) and p[1] == int(p[1])):
            return True
        else:
            return False
        
    def _vertical_cross_wall(self, p1: tuple, p2:tuple, dirc: str, step_length):
        """
        x1 = x2 = int(x1) = int(x2)
        """
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

        if self._cartesian(p1, p2) >= step_length:
            x_set = np.linspace(p2[0], p1[0], int(self._cartesian(p1, p2) / step_length) + 2)
            y_set = np.linspace(p2[1], p1[1], int(self._cartesian(p1, p2) / step_length) + 2)
            for i in range(x_set.shape[0]-1):
                if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), step_length=step_length):
                    return True

            return False

    def iscross_wall(self, p1: tuple, p2: tuple, step_length: float = 1, left_or_right = 'l', up_and_down = 'u') -> bool:        
        if p1[0] >= self.xbin or p2[0] >= self.xbin or p1[0] < 0 or p2[0] < 0:
            raise ValueError(f"{p1[0]} is not a valid value in the maze. It should belong to [0, {self.xbin}).")
        
        if p1[1] >= self.ybin or p2[1] >= self.ybin or p1[1] < 0 or p2[1] < 0:
            raise ValueError(f"{p1[1]} is not a valid value in the maze. It should belong to [0, {self.ybin}).")
        
        if p1 == p2:
            return False


        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

        #else:  # (x1 - x2)*(y1 - y2) != 0 or (x1 == x2 and x1 != len(x1)) or (y1 == y2 and y1 != len(y1))

        if self._cartesian(p1, p2) >= step_length:
            
            x_set = np.linspace(p2[0], p1[0], int(self._cartesian(p1, p2) / step_length) + 2)
            y_set = np.linspace(p2[1], p1[1], int(self._cartesian(p1, p2) / step_length) + 2)

            self._l, self._r = False, False
            self._u, self._d = False, False
            for i in range(x_set.shape[0]-1):
                if x1 == x2 and x1 == int(x1):
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), left_or_right = 'l'):
                        self._l = True
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), left_or_right = 'r'):
                        self._r = True
                    
                    if self._l and self._r:
                        return True

                    E = Edge(xbin=self.xbin, ybin=self.ybin, x=x1, y=np.min([int(y_set[i+1]), int(y_set[i])]), dirc='v')
                    E.place_wall(self.Graph)
                    if E.Wall == False:
                        self._r = self._l = False

                elif y1 == y2 and y1 == int(y1):
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), up_and_down = 'u'):
                        self._u = True
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), up_and_down = 'd'):
                        self._d = True

                    if self._u and self._d:
                        return True

                    E = Edge(xbin=self.xbin, ybin=self.ybin, x=np.min([int(x_set[i+1]), int(x_set[i])]), y=y1, dirc='h')
                    E.place_wall(self.Graph)
                    if E.Wall == False:
                        self._u = self._d = False
                    
                else:
                    if self.iscross_wall(p1=(x_set[i], y_set[i]), p2=(x_set[i+1], y_set[i+1]), step_length=step_length):
                        return True

            return False

        else:
            id1 = pvl_to_idx(x1, y1, xbin=self.xbin, ybin=self.ybin, xmax=self.xbin, ymax=self.ybin)
            id2 = pvl_to_idx(x2, y2, xbin=self.xbin, ybin=self.ybin, xmax=self.xbin, ymax=self.ybin)

            if x1 == x2 and x1 == int(x1):
                if self._is_on_point(p1) and self._is_on_point(p2):
                    P1 = Point(self.xbin, self.ybin, x1, y1)
                    P2 = Point(self.xbin, self.ybin, x2, y2)
                    E = Edge(self.xbin, self.ybin, x1, np.min([int(y1), int(y2)]), 'v')
                    P1.place_wall(self.Graph)
                    P2.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P1.EWall == False and P2.EWall == False and left_or_right == 'r':
                            return False
                        elif P1.WWall == False and P2.WWall == False and left_or_right == 'l':
                            return False
                        else:
                            return True

                elif self._is_on_point(p1) == True and self._is_on_point(p2) == False:
                    P1 = Point(self.xbin, self.ybin, x1, y1)
                    E = Edge(self.xbin, self.ybin, x1, np.min([int(y1), int(y2)]), 'v')
                    P1.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P1.EWall == False and left_or_right == 'r':
                            return False
                        elif P1.WWall == False and left_or_right == 'l':
                            return False
                        else:
                            return True
                
                elif self._is_on_point(p1) == False and self._is_on_point(p2) == True:
                    P2 = Point(self.xbin, self.ybin, x2, y2)
                    E = Edge(self.xbin, self.ybin, x1, np.min([int(y1), int(y2)]), 'v')
                    P2.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P2.EWall == False and left_or_right == 'r':
                            return False
                        elif P2.WWall == False and left_or_right == 'l':
                            return False
                        else:
                            return True         

                else: # Both of p1 and p2 are on edges instead of points.
                    if id1 == id2:
                        return False
                    else:                
                        P = Point(self.xbin, self.ybin, x1, np.max([int(y1), int(y2)]))
                        P.place_wall(self.Graph)

                        if P.EWall == False and left_or_right == 'r':
                            return False
                        if P.WWall == False and left_or_right == 'l':
                            return False
                        else:
                            return True
            
            elif y1 == y2 and y1 == int(y1):

                if self._is_on_point(p1) and self._is_on_point(p2):
                    P1 = Point(self.xbin, self.ybin, x1, y1)
                    P2 = Point(self.xbin, self.ybin, x2, y2)
                    E = Edge(self.xbin, self.ybin, np.min([int(x1), int(x2)]), y1, 'h')
                    P1.place_wall(self.Graph)
                    P2.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P1.NWall == False and P2.NWall == False and up_and_down == 'u':
                            return False
                        elif P1.SWall == False and P2.SWall == False and up_and_down == 'd':
                            return False
                        else:
                            return True

                elif self._is_on_point(p1) == True and self._is_on_point(p2) == False:
                    P1 = Point(self.xbin, self.ybin, x1, y1)
                    E = Edge(self.xbin, self.ybin, np.min([int(x1), int(x2)]), y1, 'h')
                    P1.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P1.NWall == False and up_and_down == 'u':
                            return False
                        elif P1.SWall == False and up_and_down == 'd':
                            return False
                        else:
                            return True
                
                elif self._is_on_point(p1) == False and self._is_on_point(p2) == True:
                    P2 = Point(self.xbin, self.ybin, x2, y2)
                    E = Edge(self.xbin, self.ybin, np.min([int(x1), int(x2)]), y1, 'h')
                    P2.place_wall(self.Graph)
                    E.place_wall(self.Graph)

                    if E.Wall == False:
                        return False
                    else:
                        if P2.NWall == False and up_and_down == 'u':
                            return False
                        elif P2.SWall == False and up_and_down == 'd':
                            return False
                        else:
                            return True    

                else: # Both of p1 and p2 are on edges instead of points.
                    if id1 == id2:
                        return False
                    else:                
                        P = Point(self.xbin, self.ybin, np.max([int(x1), int(x2)]), y1)
                        P.place_wall(self.Graph)

                        if P.NWall == False and up_and_down == 'u':
                            return False
                        if P.SWall == False and up_and_down == 'd':
                            return False
                        else:
                            return True             
                
            else:
                if id1 == id2:
                    return False
                
                if self._is_on_point(p1):
                    P = Point(self.xbin, self.ybin, x1, y1)
                    return True ^ P.is_passable(self.Graph, x2, y2)

                if self._is_on_point(p2):
                    P = Point(self.xbin, self.ybin, x2, y2)
                    return True ^ P.is_passable(self.Graph, x1, y1)
                
                if id1 not in self.OFGraph.Graph[id2]:
                    P = Point(self.xbin, self.ybin, np.max([int(x2), int(x1)]), np.max([int(y2), int(y1)]))
                    P.place_wall(Graph=self.Graph)

                    k, b = (y2 - y1)/(x2 - x1), ((y1 - P.y)*(x2 - P.x) - (y2 - P.y)*(x1 - P.x))/(x2 - x1)

                    if k > 0:
                        if b > 0:
                            return P.NWall or P.WWall
                        elif b == 0:
                            return True ^ P.is_passable(self.Graph, x1, y1)
                        else: # b < 0
                            return P.SWall or P.EWall
                    
                    elif k < 0:
                        if b > 0:
                            return P.NWall or P.EWall
                        elif b == 0:
                            return True ^ P.is_passable(self.Graph, x1, y1)
                        else: # b < 0
                            return P.SWall or P.WWall
                
                    # if x1 = x2 or y1 = y2, the id1 must be listed in self.OFGraph[id2], 
                    # so this case should not be taken into concerns.

                else:
                    if id1 not in self.Graph[id2]:
                        return True
                    else:
                        return False
                return False
    
    def _init_corner_graph(self):
        """
        We assumed that the shortest distance between any two points (denote as A and B) in the 
          maze must have the following forms: A -> (some turns) -> B
        
        The turns that listed here are specifically refer to points (Class Point) which satisfy
          following criteria:
          1. It should connect with 4 edges. (if only connect with 3 edges, it is on the brink of 
             the maze. If only connect with 2 edges, the point is at the corner of the maze.)
          2. It could at most connect with 2 walls.
          3. If it connects with 1 wall, it is a 'turn'. If it connects with 2 walls, the two walls
             must be placed at a 90° angle.

        The aim of this function is to find all of the turns point and connect them to build a graph.
        """
        Ps = self._find_all_corner_points()
        G = np.zeros([Ps.shape[0], Ps.shape[0]])
        for i in range(Ps.shape[0]-1):
            for j in range(i+1, Ps.shape[0]):
                if self.iscross_wall(p1 = (self.Ps[i, 0], self.Ps[i, 1]), p2 = (self.Ps[j, 0], self.Ps[j, 1])) == False:
                        if self.Points[i].is_passable(self.Graph, Ps[j, 0], Ps[j, 1]) and self.Points[j].is_passable(self.Graph, Ps[i, 0], Ps[i, 1]):
                            G[i,j] = G[j,i] = self._cartesian((self.Ps[i, 0], self.Ps[i, 1]), (self.Ps[j, 0], self.Ps[j, 1]))

        self.ConnectMat = G
        self.CornerGraph = nx.Graph(G)
        print(self.CornerGraph)
    
    def _find_all_corner_points(self):
        self.Ps = []
        self.Points = [] 
        for i in range(self.xbin+1):
            for j in range(self.ybin+1):
                P = Point(self.xbin, self.ybin, i, j)
                if P.is_corner_point(self.Graph):
                    self.Ps.append([P.x, P.y])
                    self.Points.append(P)
        self.Ps = np.array(self.Ps)
        return self.Ps

    def shortest_distance(self, p1: tuple, p2: tuple, **kwargs):
        """
        Using dijkstra algorithm to find the shortest path length between two point.
        """
        if self.iscross_wall(p1, p2) == False:
            return self._cartesian(p1, p2)

        G = cp.deepcopy(self.CornerGraph)
        nodes = len(G.nodes)

        G.add_node(nodes)
        G.add_node(nodes+1)
        for i in range(nodes):
            if self.iscross_wall(p1, (self.Ps[i, 0], self.Ps[i, 1])) == False:
                if self.Points[i].is_passable(self.Graph, p1[0], p1[1]):
                    G.add_edge(i, nodes, weight = self._cartesian(p1, (self.Ps[i, 0], self.Ps[i, 1])))
            
            if self.iscross_wall(p2, (self.Ps[i, 0], self.Ps[i, 1])) == False:
                if self.Points[i].is_passable(self.Graph, p2[0], p2[1]):
                    G.add_edge(i, nodes+1, weight = self._cartesian(p2, (self.Ps[i, 0], self.Ps[i, 1])))

        return nx.dijkstra_path_length(G=G, source=nodes, target=nodes+1, **kwargs)
    
    def shortest_path(self, p1: tuple, p2: tuple, **kwargs):
        G = cp.deepcopy(self.CornerGraph)
        nodes = len(G.nodes)

        if self.iscross_wall(p1, p2) == False:
            return [nodes, nodes+1]

        G.add_node(nodes)
        G.add_node(nodes+1)
        for i in range(nodes):
            if self.iscross_wall(p1, (self.Ps[i, 0], self.Ps[i, 1])) == False:
                if self.Points[i].is_passable(self.Graph, p1[0], p1[1]):
                    G.add_edge(i, nodes, weight = self._cartesian(p1, (self.Ps[i, 0], self.Ps[i, 1])))
            
            if self.iscross_wall(p2, (self.Ps[i, 0], self.Ps[i, 1])) == False:
                if self.Points[i].is_passable(self.Graph, p2[0], p2[1]):
                    G.add_edge(i, nodes+1, weight = self._cartesian(p2, (self.Ps[i, 0], self.Ps[i, 1])))

        return nx.dijkstra_path(G=G, source=nodes, target=nodes+1, **kwargs)

    def plot_shortest_path(self, p1: tuple, p2:tuple, ax: Axes = None, dx: float = 0.5, dy: float = 0.5,
                           figsize: tuple = (6, 6), color = 'red', linewidth = 2, **kwargs) -> Axes:
        if ax is None:
            fig = plt.figure(figsize = figsize)
            ax = plt.axes()
        
        path = self.shortest_path(p1, p2)

        if len(path) > 2:

            ax.plot([p1[0] - dx, self.Ps[path[1], 0] - dx], [p1[1] - dy, self.Ps[path[1], 1] - dy], color = color, linewidth = 2, **kwargs)
            for i in range(1, len(path)-2):
                #ax.text(self.Ps[path[i], 0] - dx, self.Ps[path[i], 1] - dx, str(path[i]))
                ax.plot([self.Ps[path[i], 0] - dx, self.Ps[path[i+1], 0] - dx], [self.Ps[path[i], 1] - dy, self.Ps[path[i+1], 1] - dy], 
                         color = color, linewidth = linewidth, **kwargs)
            ax.plot([self.Ps[path[-2], 0] - dx, p2[0] - dx], [self.Ps[path[-2], 1] - dy, p2[1] - dy], color = color, linewidth = linewidth, **kwargs)
        else: # len(path) <= 2:
            #ax.text(self.Ps[path[-2], 0] - dx, self.Ps[path[-2], 1] - dx, str(path[-2]))
            ax.plot([p1[0] - dx, p2[0] - dx], [p1[1] - dy, p2[1] - dy], color = color, linewidth = linewidth, **kwargs)

        for i in range(self.Ps.shape[0]):
            if i in path:
                ax.text(self.Ps[i, 0] - dx, self.Ps[i, 1] - dy, str(i), color = 'black')
            else:
                ax.text(self.Ps[i, 0] - dx, self.Ps[i, 1] - dy, str(i), color = 'gray')
        
        return ax


>>>>>>> bebfe805e63d3226d6196e1b8ff4c78089a1de31
