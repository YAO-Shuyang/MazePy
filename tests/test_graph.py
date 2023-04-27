from mazepy.plot.mazeprofile import MazeProfile
from mazepy.behav.graph import Graph
from mazepy.behav.element import Point, Edge

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

def test(G: Graph = None, xbin: int = 12, ybin: int = 12, p1: tuple = None, p2: tuple = None):
    if G is None:
        G = Graph(xbin=xbin, ybin=ybin, Graph={1: [2],2: [1,3,14],3: [2,15],4: [16],5: [6, 17],6: [5, 7],7: [6, 8,19],8: [7,20],9: [10,21],10: [9, 11],11: [10, 12],12: [11, 24],13: [14, 25],14: [2,13],15: [3, 16],16: [4, 15],17: [5, 29],18: [19, 30],19: [7, 18],20: [8],21: [9, 33],22: [23, 34],23: [22, 24],24: [12, 23,36],25: [13,37],26: [27],27: [26, 39],28: [29,40],29: [17, 28],30: [18, 31],31: [30, 43],32: [33, 44],33: [21, 32],34: [22, 46],35: [36,47],36: [35, 24],37: [25, 49],38: [50],39: [27, 51],40: [28,52],41: [42,53],42: [41],43: [31,44],44: [32,43, 56],45: [46, 57],46: [34, 45],47: [35],48: [ 60],49: [37, 61],50: [38, 51,62],51: [39, 50,52],52: [51,40,53],53: [41,52,65],54: [55,66],55: [54,67],56: [44],57: [45, 69],58: [70],59: [71, 60],60: [48, 59,72],61: [49, 73],62: [50],63: [64, 75],64: [63, 65],65: [64, 53],66: [54, 78],67: [55, 68],68: [67, 69],69: [68, 57],70: [58,82, 71],71: [70, 59],72: [60, 84],73: [61, 74],74: [73, 86],75: [76, 63],76: [75, 88],77: [78, 89],78: [77,66, 90],79: [80, 91],80: [79, 92],81: [82],82: [70,81, 94],83: [95, 84],84: [72, 83],85: [86, 97],86: [74,85,87],87: [86],88: [100, 76],89: [77, 101],90: [78],91: [79, 103],92: [80, 93],93: [92,105],94: [82,106],95: [83,96,107],96: [95,108],97: [85, 98],98: [97, 110],99: [111, 100],100: [99, 88],101: [89, 102],102: [101, 114],103: [91, 104],104: [103, 116],105: [93, 106,117],106: [105, 94],107: [95],108: [96, 120],109: [110,121],110: [98,109, 122],111: [99, 123],112: [113, 124],113: [112],114: [115, 102],115: [116, 114],116: [115, 104],117: [105, 129],118: [119],119: [118, 120,131],120: [108,119],121: [109, 133],122: [110],123: [111, 135],124: [125, 112],125: [124, 126],126: [125, 138],127: [128, 139],128: [127, 140],129: [117,141],130: [131,142],131: [130, 119],132: [144],133: [121, 134],134: [133, 135],135: [123,134,136],136: [135, 137],137: [136, 138],138: [137, 126],139: [127],140: [141, 128],141: [129, 140],142: [130,143],143: [142,144],144: [132,143]})
    
    if p1 is None:
        p1 = (np.random.rand()*xbin, np.random.rand()*ybin)
    if p2 is None:
        p2 = (np.random.rand()*xbin, np.random.rand()*ybin)

    fig = plt.figure(figsize=(6,6))
    ax = plt.axes()
    ax.set_aspect('equal')

    occu_map = np.zeros(xbin*ybin)*np.nan
    occu_map[0] = 0
    Maze = MazeProfile(xbin=xbin, ybin=ybin, Graph=G.Graph, occu_map=occu_map)
    ax = Maze.DrawMazeProfile(ax=ax, linewidth=3, color = 'black')
    ax.set_title(f"P1({round(p1[0], 3), round(p1[1], 3)})\n P2({round(p2[0], 3), round(p2[1], 3)})")
    t1 = time.time()
    G.plot_shortest_path(p1, p2, ax=ax)
    print(f'Time for 1 trial: {time.time()-t1} s')
    plt.show()


if __name__ == "__main__":
    t1 = time.time()
    G = Graph(xbin=12, ybin=12, Graph={1: [2, 13],2: [1],3: [4, 15],4: [3, 5],5: [4, 6],6: [5, 7, 18],7: [6, 8],8: [7],9: [10, 21],10: [9, 11],11: [10, 12],12: [11, 24],13: [1, 14, 25],14: [13, 26],15: [3, 27],16: [17, 28],17: [16, 18, 29],18: [6, 17],19: [20, 31],20: [19, 21],21: [9, 20],22: [23, 34],23: [22, 24],24: [12, 23,36],25: [13],26: [14, 27],27: [26, 15],28: [16],29: [17, 30],30: [29, 31, 42],31: [30, 19],32: [33, 44],33: [32, 34],34: [22, 33],35: [36],36: [35, 24],37: [38, 49],38: [37, 39],39: [40, 38, 51],40: [39],41: [42],42: [41, 30],43: [55],44: [32, 45],45: [44, 46],46: [45, 47],47: [46, 48],48: [47, 60],49: [37, 61],50: [51, 62],51: [39, 50,52],52: [51],53: [54],54: [53, 55,66],55: [43, 54,67],56: [57, 68],57: [56, 58],58: [57, 59],59: [58, 60],60: [48, 59],61: [49, 73],62: [50, 74],63: [64, 75],64: [63, 65],65: [64, 66],66: [54, 65],67: [55, 79],68: [56, 69],69: [68, 70],70: [69, 71],71: [70, 72],72: [71, 84],73: [61, 85],74: [62, 75],75: [74, 63],76: [77, 88],77: [76, 89],78: [79, 90],79: [78, 67],80: [81, 92],81: [80, 82],82: [81, 94],83: [95, 84],84: [72, 83, 96],85: [73, 97],86: [98],87: [88, 99],88: [87, 76],89: [77, 101],90: [78, 91],91: [90, 103],92: [80, 104],93: [105],94: [82, 95,106],95: [83, 94],96: [84],97: [85, 98, 109],98: [97, 86],99: [87, 100],100: [99, 112],101: [89, 102],102: [101, 114],103: [91, 104],104: [103, 92],105: [93, 106],106: [105, 94],107: [108, 119],108: [107, 120],109: [97, 110,121],110: [109, 122],111: [112, 123],112: [100, 111],113: [114, 125],114: [113, 102],115: [116, 127],116: [115, 117],117: [116, 129],118: [119, 130],119: [118, 107],120: [108],121: [109, 133],122: [110, 123],123: [111, 122],124: [125, 136],125: [124, 113],126: [127, 138],127: [115, 126],128: [129, 140],129: [128, 117,141],130: [118, 131,142],131: [130, 132],132: [131, 144],133: [121, 134],134: [133, 135],135: [134],136: [124, 137],137: [136, 138],138: [137, 126],139: [140],140: [139, 128],141: [129, 142],142: [130, 141,143],143: [142],144: [132]})

    print(f'init time: {time.time()-t1} s')

    #test(G=G, xbin=12, ybin=12, p1=(G.Ps[20,0], G.Ps[20,1]), p2=(G.Ps[16,0], G.Ps[16,1]))

    for i in range(10):
        test(G=G, xbin=12, ybin=12)
