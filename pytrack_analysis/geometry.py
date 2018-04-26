import numpy as np
import pandas as pd
import platform, subprocess
import os, sys
import cv2
from io import StringIO
from pytrack_analysis.cli import colorprint, flprint, prn
from pytrack_analysis.image_processing import VideoCaptureAsync, VideoCapture, match_templates, get_peak_matches, preview

NAMESPACE = 'geometry'

"""
Returns angle between to given points centered on pt1 (GEOMETRY)
"""
def get_angle(pt1, pt2, flipy=False):
    dx = pt2[0]-pt1[0]
    if flipy:
        dy = pt1[1]-pt2[1]
    else:
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

class Arena(object):
    def __init__(self, x, y, scale, c, layout=None):
        self.x = x
        self.y = y
        self.r = scale * 25.
        self.r0 = scale * .1
        self.scale = scale
        self.rotation = 0.
        self.substr = ['yeast', 'sucrose']
        self.spots = self.get_rings((scale, 10., 0., np.pi/3., 0), (scale, 20., np.pi/6., np.pi/3., 1))

    def get_dict(self):
        return {'x': self.x, 'y': self.y, 'radius': self.r, 'scale': self.scale, 'rotation': self.rotation}

    def move_to(self, x, y):
        dx = x - self.x
        dy = y - self.y
        self.x = x
        self.y = y
        for spot in self.spots:
            spot.move_by(dx, dy)

    def rotate_by(self, value):
        self.rotation += value
        self.rotation = round(self.rotation, 2)
        self.spots = self.get_rings((self.scale, 10., 0.+self.rotation, np.pi/3., 0), (self.scale, 20., np.pi/6.+self.rotation, np.pi/3., 1))

    def scale_by(self, value):
        self.scale += value
        self.scale = round(self.scale, 5)
        if self.scale < 0.0:
            self.scale = 0.00
        self.r = self.scale * 25.
        self.r0 = self.scale * .5
        self.spots = self.get_rings((self.scale, 10., 0.+self.rotation, np.pi/3., 0), (self.scale, 20., np.pi/6.+self.rotation, np.pi/3., 1))

    def get_rings(self, *args):
        """
        takes tuples: (scale, distance, offset, interval, substrate_move)
        """
        out = []
        for arg in args:
            sc = arg[0]
            r = sc * arg[1]
            t = arg[2]
            w = arg[3]
            sm = arg[4]
            angles = np.arange(t, t+2*np.pi, w)
            xs, ys = r * np.cos(angles), r * np.sin(angles)
            for i, (x, y) in enumerate(zip(xs, ys)):
                out.append(Spot(x+self.x, y+self.y, sc * 1.5, self.substr[(i+sm)%2]))
        return out


class Spot(object):
    def __init__(self, x, y, r, s):
        self.x = x
        self.y = y
        self.r = r
        self.substrate = s

    def move_by(self, dx, dy):
        self.x += dx
        self.y += dy

    def move_to(self, x, y):
        self.x = x
        self.y = y

    def toggle_substrate(self):
        s = {'yeast': 'sucrose', 'sucrose': 'yeast'}
        self.substrate = s[self.substrate]

def detect_geometry(_fullpath):
    setup = os.path.basename(_fullpath).split('_')[0]
    video = VideoCapture(_fullpath, 0)
    img = video.get_average(100) #### takes average of 100 frames
    video.stop()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### this is for putting more templates
    cv2.imwrite('res.png',img)

    """
    Get arenas
    """
    thresh = 0.8
    arenas = []
    while len(arenas) != 4:
        img_rgb =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) ### this is for coloring
        ### Template matching function
        loc, vals, w = match_templates(img, 'arena', setup, thresh)
        patches = get_peak_matches(loc, vals, w, img_rgb)
        arenas = patches
        ### Required to have 4 arenas detected, lower threshold if not matched
        if len(arenas) < 4:
            thresh -= 0.05
            thresh = round(thresh,2)
            print('Not enough arenas detected. Decrease matching threshold to {}.'.format(thresh))
        elif len(arenas) == 4:
            print('Detected 4 arenas. Exiting arena detection.')
            for pt in arenas:
                ept = (int(round(pt[0]+w/2)), int(round(pt[1]+w/2)))
                cv2.circle(img_rgb, ept, int(w/2), (0,255,0), 1)
                cv2.circle(img_rgb, ept, 1, (0,255,0), 2)
        #preview(img_rgb)
    """ Geometry correction algorithm (not used)
    for i, arena in enumerate(arenas):
        print('(x, y) = ({}, {})'.format(arena[0], arena[1]))
    for i, j in zip([0, 1, 3, 2], [1, 3, 2, 0]):
        print('distance ({}-{}) = {}'.format(i, j, get_distance(arenas[i], arenas[j])))
        print('Angle: {}'.format(get_angle(arenas[i], arenas[j], flipy=True)))
    for i, j, k in zip([3, 2, 1, 0], [1, 0, 0, 1], [2, 3, 3, 2]): ### i is anchor
        for m in range(4):
            if m not in [i, j, k]:
                print(m)
        da = [arenas[j][0]-arenas[i][0], arenas[j][1]-arenas[i][1]]
        pta = [arenas[k][0]+da[0], arenas[k][1]+da[1]]
        db = [arenas[k][0]-arenas[i][0], arenas[k][1]-arenas[i][1]]
        ptb = [arenas[j][0]+db[0], arenas[j][1]+db[1]]
        if get_distance(pta, ptb) < 5:
            pt = (int((pta[0]+ptb[0])/2 + w/2), int((pta[1]+ptb[1])/2 + w/2))
            cv2.circle(img_rgb, pt, int(w/2), (255,0,255), 1)
            cv2.circle(img_rgb, pt, 1, (255,0,255), 2)
    """
    preview(img_rgb, title='Preview arena', topleft='Threshold: {}'.format(thresh))



    """
    Get spots
    """
    labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
    for ia, arena in enumerate(arenas):
        arena_img = img[arena[1]:arena[1]+w, arena[0]:arena[0]+w]
        spots = []
        thresh = 0.99
        while len(spots) < 6:
            img_rgb =  cv2.cvtColor(arena_img,cv2.COLOR_GRAY2RGB) ### this is for coloring
            ### Template matching function
            loc, vals, ws = match_templates(arena_img, 'yeast', setup, thresh)
            patches = get_peak_matches(loc, vals, ws, img_rgb, arena=arena)
            spots = patches
            ### Required to have 6 yeast spots detected, lower threshold if not matched
            if len(spots) < 6:
                thresh -= 0.01
                thresh = round(thresh,2)
                print('Not enough yeast spots detected. Decrease matching threshold to {}.'.format(thresh))
            else:
                print('Detected 6 yeast spots. Exiting spot detection.')
                for pt in spots:
                    ept = (int(round(pt[0]+ws/2)), int(round(pt[1]+ws/2)))
                    cv2.circle(img_rgb, ept, int(ws/2), (0,165,255), 1)
                    cv2.circle(img_rgb, ept, 1, (0,165,255), 1)
        preview(img_rgb, title='Preview spots', topleft='Arena: {}, threshold: {}'.format(labels[ia], thresh))

    """
    yeasts = [os.path.join('..', 'media', 'templates', setup, _file) for _file in os.listdir(os.path.join('..', 'media', 'templates', setup)) if 'yeast' in _file]
    templates = [cv2.imread(_file,0) for _file in yeasts]
    ws, h = templates[0].shape[::-1]
    res = [cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) for template in templates]
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
        #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + w), (0,0,255), 1)
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
        cv2.circle(img_rgb, ept, 1, (0,165,255), 1)
    #print(loc)
    """
    return arenas


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
