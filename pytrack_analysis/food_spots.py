import numpy as np
import pandas as pd
from pytrack_analysis.cli import colorprint, flprint, prn
from pytrack_analysis.geometry import get_angle, get_distance, rot

"""
Class for defining food spots
"""
class Spot(object):
    def __init__(self, _x, _y, _r, _s):
        self.x = _x
        self.y = _y
        self.r = _r
        self.substrate = _s

class SpotCollection(object):
    def __init__(self):
        self.spots = []
    def add(self, _spot):
        self.spots.append(_spot)
    def get(self, _index):
        if type(_index) is int:
            return self.spots[_index]
    def __getitem__(self, key):
        return self.spots[key]

"""
Returns food spots data (DATAIO)
"""
def get_food(filenames, arenas):
    prn(__name__)
    flprint("loading food spots data...")
    data = []
    skip = False
    for each in filenames:
        np.warnings.filterwarnings("ignore")
        eachdata = np.loadtxt(each)
        np.warnings.filterwarnings("default")
        if eachdata.shape[0] == 0:
            skip = True
        elif len(eachdata.shape) == 1:
            eachdata = np.reshape(eachdata, (1, 2))
        ## load data
        data.append(eachdata)
    if not skip:
        data = validate_food(data, arenas)
    else:
        data = None
    if data is None:
        colorprint("ERROR: no inner spots detected. please redo food spot detection in Bonsai.", color='error')
    else:
        colorprint("done.", color='success')
    return data

"""
Returns new spots (DATAIO)
"""
def get_new_spots(spots, mindist, left=0):
    new_spot = []
    for each_spot in spots:
        mat = rot(120, in_degrees=True)
        vec = np.array([each_spot[0], each_spot[1]])
        out = np.dot(mat, vec)
        x, y = out[0], out[1]
        dists = [get_distance((spot[0], spot[1]), (x,y)) for spot in spots]
        for each_distance in dists:
            if each_distance < mindist:
                mat = rot(-120, in_degrees=True)
                vec = np.array([each_spot[0], each_spot[1]])
                out = np.dot(mat, vec)
                x, y = out[0], out[1]
        new_spot.append([x, y])
        if left > 1:
            mat = rot(-120, in_degrees=True)
            vec = np.array([each_spot[0], each_spot[1]])
            out = np.dot(mat, vec)
            x, y = out[0], out[1]
            new_spot.append([x, y])
    return new_spot

"""
Validate food spots (DATAIO)
"""
def validate_food(spots, arenas, VERBOSE=False):
    fix_outer = 260
    spot_radius = 12.815
    pxmm = 8.543
    for ix, spots_arena in enumerate(spots):
        if VERBOSE: flprint("Loading food spots for arena {}...".format(ix))
        p0 = np.array([arenas[ix].x, arenas[ix].y])        # arena center
        f0 = p0 - fix_outer             # frame origin

        all_spots = []

        n_inner, n_outer = 3, 3 # counter for inner and outer spots found

        ### spots for inner and outer triangle
        inner, outer = [], []

        for each_spot in spots_arena:
            spot = each_spot - p0 + f0  ### vector centered around arena center
            d = get_distance((0, 0), spot)
            a = get_angle((0, 0), spot)

            ### validate existing spots
            if d > 30. and d < 212.5:
                all_spots.append([spot[0], spot[1], 0])
                if d < 120.:        ### inner spots
                    n_inner -= 1
                    inner.append([spot[0], spot[1], a])
                    if VERBOSE: print('inner:', spot[0], spot[1], d, np.degrees(a))
                else:               ### outer spots
                    n_outer -= 1
                    outer.append([spot[0], spot[1], a])
                    if VERBOSE: print('outer:', spot[0], spot[1], d, np.degrees(a))
            ### removal
            else:
                if VERBOSE: print("Removed: ", spot[0], spot[1], d)

        ### Check whether existing inner spots are of right number
        if n_inner < 0:
            #print("Too many inner spots. Removing {} spots.".format(-n_inner))
            min_dis = fix_outer
            ### remove as many spots as needed based on avg distance to all spots
            for each in range(-n_inner):
                for ispot, spot in enumerate(inner):
                    mean_dis = np.mean([get_distance((spot[0], spot[1]), (other[0], other[1])) for other in inner])
                    if mean_dis < min_dis:
                        min_dis = mean_dis
                        remove = ispot
                all_spots.remove([inner[ispot][0], inner[ispot][1], 0])
                del inner[ispot]

        ### Check for translation (needs to be after removal of extra spots)
        if len(inner) == 3:
            tx, ty = 0, 0
            for each_spot in inner:
                tx += each_spot[0]
                ty += each_spot[1]
            tx /= len(inner)
            ty /= len(inner)
        elif len(inner) == 0:
            return None
        elif len(inner) < 3:
            tx, ty = 0, 0
            for each_spot in inner:
                dr = get_distance((0, 0), (each_spot[0], each_spot[1])) - 10.*pxmm #### This is 10 mm
                tx += dr * np.cos(get_angle((0, 0), (each_spot[0], each_spot[1])))
                ty += dr * np.sin(get_angle((0, 0), (each_spot[0], each_spot[1])))
            tx /= len(inner)
            ty /= len(inner)
        if VERBOSE: print("Translation detected: ({}, {})".format(tx, ty))
        ### Correcting for translation
        for spot in inner:
            spot[0] -= tx
            spot[1] -= ty
        for spot in outer:
            spot[0] -= tx
            spot[1] -= ty
        for spot in all_spots:
            spot[0] -= tx
            spot[1] -= ty

        if n_inner > 0:
            if VERBOSE: print("Too few inner spots. Missing {} spots.".format(n_inner))
            ### getting new spots by means of rotation
            near_new = get_new_spots(inner, 80, left=n_inner)
            ### overlapping spots get averaged to one
            if get_distance(near_new[0], near_new[1]) < 40.:
                near_new = [[0.5*(near_new[0][0] + near_new[1][0]), 0.5*(near_new[0][1] + near_new[1][1])]]
            ### add new one to list of all spots
            for spot in near_new:
                inner.append([spot[0], spot[1], get_angle((0,0), spot)])
                all_spots.append([spot[0], spot[1], 1])

        ### Check whether existing outer spots are of right number
        if n_outer < 0:
            if VERBOSE: print("Too many outer spots. Removing {} spots.".format(-n_outer))
            min_dis = 260.
            ### remove as many spots as needed based on avg distance to all spots
            for each in range(-n_outer):
                for ispot, spot in enumerate(outer):
                    mean_dis = np.mean([get_distance((spot[0], spot[1]), (other[0], other[1])) for other in inner])
                    if mean_dis < min_dis:
                        min_dis = mean_dis
                        remove = ispot
                all_spots.remove([inner[ispot][0], inner[ispot][1], 0])
                del inner[ispot]

        if n_outer > 0:
            if VERBOSE: print("Too few outer spots. Missing {} spots.".format(n_outer))
            if n_outer < 3:
                far_new = get_new_spots(outer, 250, left=n_outer)
                ### overlapping spots get averaged to one
                if get_distance(far_new[0], far_new[1]) < 40.:
                    far_new = [[0.5*(far_new[0][0] + far_new[1][0]), 0.5*(far_new[0][1] + far_new[1][1])]]
                ### add new one to list of all spots
                for spot in far_new:
                    all_spots.append([spot[0], spot[1], 1])
            else:
                for spot in inner:
                    x, y, a = spot[0], spot[1], spot[2]
                    nx = x + 10.*pxmm * np.cos(a)
                    ny = y + 10.*pxmm * np.sin(a)
                    mat = rot(90, in_degrees=True)
                    vec = np.array([nx, ny])
                    out = np.dot(mat, vec)
                    all_spots.append([out[0], out[1], 1])

        ### Adding sucrose by rotating yeast positions
        sucrose = []
        for spot in all_spots:
            mat = rot(60, in_degrees=True)
            vec = np.array([spot[0], spot[1]])
            out = np.dot(mat, vec)
            sucrose.append([out[0], out[1], 2])
        # add them all to list
        for each_spot in sucrose:
            all_spots.append(each_spot)

        ### return all spots for this arena into list
        for each_spot in all_spots:
            if each_spot[2] < 2:
                substr = 'yeast'
            if each_spot[2] == 2:
                substr = 'sucrose'
            arenas[ix].spots.add(Spot(each_spot[0], each_spot[1], spot_radius, substr))
        spots[ix] = np.array(all_spots)
        if VERBOSE: print("found", len(spots[ix]), "food spots.")
    flprint("found a total of {} spots...".format(sum([len(each) for each in spots])))
    return spots
