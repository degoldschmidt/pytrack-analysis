import numpy as np
import pandas as pd
import cv2
import platform, subprocess
import os
from pytrack_analysis.cli import colorprint, flprint, prn

NAMESPACE = 'geometry'

"""
Returns angle between to given points centered on pt1 (GEOMETRY)
"""
def get_angle(pt1, pt2):
    dx = pt2[0]-pt1[0]
    dy = pt2[1]-pt1[1]
    return np.arctan2(dy,dx)

"""
Returns distance between to given points (GEOMETRY)
"""
def get_distance(pt1, pt2):
    dx = pt1[0]-pt2[0]
    dy = pt1[1]-pt2[1]
    return np.sqrt(dx**2 + dy**2)

"""
Returns rotation matrix for given angle in two dimensions (GEOMETRY)
"""
def rot(angle, in_degrees=False):
    if in_degrees:
        rads = np.radians(angle)
        return np.array([[np.cos(rads), -np.sin(rads)],[np.sin(rads), np.cos(rads)]])
    else:
        return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

"""
Returns rotated vector for given angle in two dimensions (GEOMETRY)
"""
"""
def rot(angle, in_degrees=False):
    if in_degrees:
        rads = np.radians(angle)
        return np.array([[np.cos(rads), -np.sin(rads)],[np.sin(rads), np.cos(rads)]])
    else:
        return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
"""

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

def detect_geometry(_fullpath):
    cap = cv2.VideoCapture(_fullpath)
    setup = os.path.basename(_fullpath).split('_')[0]
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('res.png',img)
    yeasts = [os.path.join('..', 'media', 'templates', setup, _file) for _file in os.listdir(os.path.join('..', 'media', 'templates', setup)) if 'yeast' in _file]
    arenas = [os.path.join('..', 'media', 'templates', setup, _file) for _file in os.listdir(os.path.join('..', 'media', 'templates', setup)) if 'arena' in _file]
    print(yeasts, arenas)
    templates = [cv2.imread(_file,0) for _file in yeasts]
    templates_arena = [cv2.imread(_file,0) for _file in arenas]
    w, h = templates[0].shape[::-1]
    w_arena, h_arena = templates_arena[0].shape[::-1]
    print(w_arena, h_arena)
    res = [cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) for template in templates]
    res_arena = [cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) for template in templates_arena]

    img_rgb =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    """
    Get arenas
    """
    arena_threshold = 0.85
    loc = None
    for r in res_arena:
        if loc is None:
            loc = list(np.where( r >= arena_threshold ))
        else:
            temp = list(np.where( r >= arena_threshold ))
            loc[0] = np.append(loc[0], temp[0])
            loc[1] = np.append(loc[1], temp[1])
    patches = []
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w_arena, pt[1] + w_arena), (0,0,255), 2)
        if len(patches) == 0:
            patches.append([pt])
        else:
            flagged = False
            for i, each_patch in enumerate(patches):
                for eachpt in each_patch:
                    if abs(eachpt[0]-pt[0]) < w_arena/4 and abs(eachpt[1]-pt[1]) < w_arena/4:
                        patches[i].append(pt)
                        break
                    elif abs(eachpt[0]-pt[0]) < w_arena and abs(eachpt[1]-pt[1]) < w_arena:
                        flagged = True
            if all([pt not in each_patch for each_patch in patches]) and not flagged:
                patches.append([pt])
    arenas = []
    for each_patch in patches:
        tis = np.array(each_patch)
        arenas.append(np.mean(tis, axis=0))
    for pt in arenas:
        ept = (int(round(pt[0]+w_arena/2)), int(round(pt[1]+w_arena/2)))
        cv2.circle(img_rgb, ept, int(w_arena/2), (0,255,0), 2)
        cv2.circle(img_rgb, ept, 1, (0,255,0), 2)

    """
    Get spots
    """
    threshold = 0.9
    loc = None
    for r in res:
        if loc is None:
            loc = list(np.where( r >= threshold ))
        else:
            temp = list(np.where( r >= threshold ))
            loc[0] = np.append(loc[0], temp[0])
            loc[1] = np.append(loc[1], temp[1])

    loc = tuple(loc)
    patches = []
    for pt in zip(*loc[::-1]):
        #cv2.rectangle(img_rgb, pt, (pt[0] + w_arena, pt[1] + w_arena), (0,0,255), 2)
        if len(patches) == 0:
            patches.append([pt])
        else:
            for i, each_patch in enumerate(patches):
                for eachpt in each_patch:
                    if abs(eachpt[0]-pt[0]) < w and abs(eachpt[1]-pt[1]) < h:
                        patches[i].append(pt)
                        break
            if all([pt not in each_patch for each_patch in patches]):
                patches.append([pt])
    spots = []
    for each_patch in patches:
        tis = np.array(each_patch)
        tismean = np.mean(tis, axis=0)
        inarena = [(abs(arena[0]+w_arena/2-tismean[0]-w/2) < w_arena/2 and abs(arena[1]+w_arena/2-tismean[1]-w/2) < w_arena/2) for arena in arenas]
        if any(inarena):
            spots.append(tismean)
    for pt in spots:
        ept = (int(round(pt[0]+w/2)), int(round(pt[1]+w/2)))
        cv2.circle(img_rgb, ept, int(w/2), (0,165,255), 1)
        cv2.circle(img_rgb, ept, 1, (0,165,255), 2)
    #print(loc)

    preview = cv2.resize(img_rgb, (704, 700))
    cv2.imshow('preview geometry (press any key to continue)',preview)
    if platform.system() == 'Darwin':
        tmpl = 'tell application "System Events" to set frontmost of every process whose unix id is {} to true'
        script = tmpl.format(os.getpid())
        output = subprocess.check_call(['/usr/bin/osascript', '-e', script])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return arenas, spots


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
