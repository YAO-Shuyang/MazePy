'''
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
'''

import numpy as np
from dataclasses import dataclass
from behav.gridbin import GridBasic


def isNorthBorder(BinID, xbin = 12, ybin = 12):
    if (BinID-1) // xbin == ybin - 1:
        return True
    else:
        return False

def isEastBorder(BinID, xbin = 12):
    if BinID % xbin == 0:
        return True
    else:
        return False

def isWestBorder(BinID, xbin = 12):
    if BinID % xbin == 1:
        return True
    else:
        return False

def isSouthBorder(BinID, xbin = 12):
    if BinID <= xbin:
        return True
    else:
        return False

@dataclass
class OpenFieldGraph(GridBasic):
    '''
    Generate the graph of open field. The basic information of the grids should be given.
    '''
    Graph:dict = {}
    
    def generate_graph(self):
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
        return self.Graph
    

class DIYGraph(OpenFieldGraph):
    '''
    A do-it-yourself graph via GUI
    '''

    def setup_gui(self):
        self.generate_graph()
        