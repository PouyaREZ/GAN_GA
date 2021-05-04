from __future__ import division
import math as mp

def PairedDistance(Pair_1, Pair_2):
    ''' This function takes in two lists of coordinate pairs and calculates the
        straight line distance between them.
    '''
    sq1 = (Pair_1[0] - Pair_2[0])*(Pair_1[0] - Pair_2[0])
    sq2 = (Pair_1[1] - Pair_2[1])*(Pair_1[1] - Pair_2[1])
    return mp.sqrt(sq1+sq2)
