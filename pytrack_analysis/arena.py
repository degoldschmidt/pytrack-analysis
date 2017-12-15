import numpy as np
import pandas as pd
from pytrack_analysis.cli import colorprint, flprint, prn
from pytrack_analysis.food_spots import SpotCollection

"""
This is a class for arena
"""
class Arena(object):
    def __init__(self, _x, _y, _r):
        self.x = _x
        self.y = _y
        self.r = _r
        self.spots = SpotCollection()

    def set_scale(self, _val):
        self.pxmm = self.r/_val

class ArenaCollection(object):
    def __init__(self):
        self.arenas = []
        self.labels = {'topleft': 0, 'topright': 1, 'bottomleft': 2, 'bottomright': 3}
    def add(self, _arena):
        self.arenas.append(_arena)
    def get(self, _index):
        if type(_index) is int:
            return self.arenas[_index]
        if type(_index) is str:
            return self.arenas[self.labels[_index.lower()]]
    def __getitem__(self, key):
        return self.arenas[key]
    def set_scale(self, _val):
        for each_arena in self.arenas:
            each_arena.set_scale(_val)

"""
Returns list of raw data for filenames (DATAIO)
"""
def get_geom(filename, labels):
    prn(__name__)
    flprint("loading geometry data...")
    data = pd.read_csv(filename, sep="\s+")
    data = np.array(data.loc[len(data.index)-1])
    arenas = ArenaCollection()
    for each_arena in range(4):
        index = 9 + 3*each_arena
        radius = 0.25*(data[index]+data[index+1]) + 30 ### radius = half of mean of major and minor
        arenas.add(Arena(data[2*each_arena], data[2*each_arena+1], radius))
    colorprint("done.", color='success')
    return arenas
