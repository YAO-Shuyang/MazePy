from mazepy.plot.mazeprofile import MazeProfile
from mazepy.behav.graph import Graph
from mazepy.behav.element import Point, Edge

import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    
    sample_graph = {1: [2],2: [1,3,14],3: [2,15],4: [16],5: [6, 17],6: [5, 7],7: [6, 8,19],8: [7,20],9: [10,21],10: [9, 11],11: [10, 12],12: [11, 24],13: [14, 25],14: [2,13],15: [3, 16],16: [4, 15],17: [5, 29],18: [19, 30],19: [7, 18],20: [8],21: [9, 33],22: [23, 34],23: [22, 24],24: [12, 23,36],25: [13,37],26: [27],27: [26, 39],28: [29,40],29: [17, 28],30: [18, 31],31: [30, 43],32: [33, 44],33: [21, 32],34: [22, 46],35: [36,47],36: [35, 24],37: [25, 49],38: [50],39: [27, 51],40: [28,52],41: [42,53],42: [41],43: [31,44],44: [32,43, 56],45: [46, 57],46: [34, 45],47: [35],48: [ 60],49: [37, 61],50: [38, 51,62],51: [39, 50,52],52: [51,40,53],53: [41,52,65],54: [55,66],55: [54,67],56: [44],57: [45, 69],58: [70],59: [71, 60],60: [48, 59,72],61: [49, 73],62: [50],63: [64, 75],64: [63, 65],65: [64, 53],66: [54, 78],67: [55, 68],68: [67, 69],69: [68, 57],70: [58,82, 71],71: [70, 59],72: [60, 84],73: [61, 74],74: [73, 86],75: [76, 63],76: [75, 88],77: [78, 89],78: [77,66, 90],79: [80, 91],80: [79, 92],81: [82],82: [70,81, 94],83: [95, 84],84: [72, 83],85: [86, 97],86: [74,85,87],87: [86],88: [100, 76],89: [77, 101],90: [78],91: [79, 103],92: [80, 93],93: [92,105],94: [82,106],95: [83,96,107],96: [95,108],97: [85, 98],98: [97, 110],99: [111, 100],100: [99, 88],101: [89, 102],102: [101, 114],103: [91, 104],104: [103, 116],105: [93, 106,117],106: [105, 94],107: [95],108: [96, 120],109: [110,121],110: [98,109, 122],111: [99, 123],112: [113, 124],113: [112],114: [115, 102],115: [116, 114],116: [115, 104],117: [105, 129],118: [119],119: [118, 120,131],120: [108,119],121: [109, 133],122: [110],123: [111, 135],124: [125, 112],125: [124, 126],126: [125, 138],127: [128, 139],128: [127, 140],129: [117,141],130: [131,142],131: [130, 119],132: [144],133: [121, 134],134: [133, 135],135: [123,134,136],136: [135, 137],137: [136, 138],138: [137, 126],139: [127],140: [141, 128],141: [129, 140],142: [130,143],143: [142,144],144: [132,143]}
    p1, p2 = (3.5, 2.5), (9.5, 11.5)
    t1 = time.time()
    G = Graph(xbin=12, ybin=12, Graph=sample_graph)

    distance, p1_list, p2_list = G.shortest_distance(p1, p2)
    print(distance)
    
    print(time.time()-t1)

    Gg = G.ConnectMat
    Ps = G.Ps

    fig = plt.figure(figsize=(6,6))
    ax = plt.axes()
    ax.set_aspect('equal')
    #ax.plot([p1[0]-0.5, p2[0]-0.5], [p1[1]-0.5, p2[1]-0.5])

    for i in range(Gg.shape[0]-1):
        for j in range(i+1, Gg.shape[0]):
            if Gg[i, j] != 0:
                ax.plot([Ps[i, 0] - 0.5, Ps[j, 0] - 0.5], [Ps[i, 1] - 0.5, Ps[j, 1] - 0.5], ls = ':', linewidth = 4)
                #ax.text((Ps[i, 0] + Ps[j, 0])/2 - 0.5, (Ps[i, 1] + Ps[j, 1])/2 - 0.5, str(round(Gg[i, j], 1)))


    for a in p1_list:
        ax.plot([p1[0]-0.5, a[0]-0.5], [p1[1]-0.5, a[1]-0.5], color = 'black')

    for a in p2_list:
        ax.plot([p2[0]-0.5, a[0]-0.5], [p2[1]-0.5, a[1]-0.5], color = 'black')

    occu_map = np.zeros(144)*np.nan
    occu_map[0] = 0
    Maze = MazeProfile(xbin=12, ybin=12, Graph=sample_graph, occu_map=occu_map, ax = ax, color = 'gray', linewidth = 2)
    """

    import networkx as nx

    Mat = np.array([[0, 1, 4, 3, 0],
                    [1, 0, 0, 2, 1],
                    [4, 0, 0, 0, 0],
                    [3, 2, 0, 0, 6],
                    [0, 1, 0, 6, 0]])
    G = nx.Graph(Mat)
    print(nx.shortest_path_length(G, 0, 3))
    pos=nx.circular_layout(G)
    nx.draw(G,pos = pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

    

    G.add_node(5)
    G.add_edge(0, 5, weight = 0.5)

    pos=nx.circular_layout(G)
    nx.draw(G,pos = pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()
    """