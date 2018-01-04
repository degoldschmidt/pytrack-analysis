import numpy as np
import pandas as pd
from pytrack_analysis.cli import colorprint, flprint, prn
from pytrack_analysis.food_spots import SpotCollection

"""
This is a class for arena
"""
class Arena(object):
    def __init__(self, _x, _y, _r, _o, _l):
        self.x = _x
        self.y = _y
        self.r = _r
        self.outer = _o
        self.name = _l
        self.spots = SpotCollection()

    def set_rscale(self, _val):
        self.pxmm = self.r/_val

    def set_scale(self, _val):
        self.pxmm = _val
        self.rr = self.r / _val
        self.ro = self.outer / _val
        for each_spot in self.spots:
            each_spot.rx = each_spot.x / _val
            each_spot.ry = each_spot.y / _val
            each_spot.rr = each_spot.r / _val

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
    def set_rscale(self, _val):
        for each_arena in self.arenas:
            each_arena.set_rscale(_val)
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
    labels = list(arenas.labels.keys())
    for each_arena in range(4):
        index = 9 + 3*each_arena
        radius = 0.25*(data[index]+data[index+1]) + 30 ### radius = half of mean of major and minor
        outer_radius = 260
        arenas.add(Arena(data[2*each_arena], data[2*each_arena+1], radius, outer_radius, labels[each_arena]))
    colorprint("done.", color='success')
    return arenas
