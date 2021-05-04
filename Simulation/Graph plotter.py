# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:12 2018

Applet for sketching the graph given the Sites_Toy and Connections_Toy CSVs.

@Author: Author
"""

# Importing libs
import networkx as nx
import pandas as pd
import numpy as np


# Importing coordinates' csv, defining the node numbers (names), coords, and positions (as a dict)
coords_csv = pd.read_csv('Sites_Toy.csv', header = 0)
nodes = list(coords_csv['Site Number'])
coords = list(zip(nodes, coords_csv['X Coordinate'], coords_csv['Y Coordinate']))
pos = {x:(y,z) for x, y, z in coords}

# Importing connections' csv, defining labels (as a dict)
connects_csv = pd.read_csv('Connections_Toy.csv', header = 0, index_col = 0)
labels = {x:y for x, y in zip(nodes, connects_csv.index[:])}

# Defining the graph based on the connections' matrix
Adj_Matrix = np.matrix(connects_csv)
g = nx.from_numpy_matrix(Adj_Matrix)
# Sketching the graph based on the positions and labels defined above
nx.draw(g, pos = pos, labels = labels, node_color='teal', font_family = 'Garamond', font_weight = 'bold')