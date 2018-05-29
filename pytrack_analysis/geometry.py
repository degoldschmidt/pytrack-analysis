import numpy as np
import pandas as pd
import platform, subprocess
import os, sys
import os.path as op
import cv2
from io import StringIO
from pytrack_analysis.cli import colorprint, flprint, prn
from pytrack_analysis.image_processing import VideoCaptureAsync, VideoCapture, match_templates, get_peak_matches, preview
from pytrack_analysis.yamlio import read_yaml, write_yaml

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

def detect_geometry(_fullpath, _timestr, onlyIm=False):
    setup = op.basename(_fullpath).split('_')[0]
    video = VideoCapture(_fullpath, 0)
    dir = op.dirname(_fullpath)
    if not op.isdir(op.join(dir, 'pytrack_res', 'arena')):
        os.mkdir(op.join(dir, 'pytrack_res', 'arena'))
    outfile = op.join(dir, 'pytrack_res','arena', setup+'_arena_' +_timestr+'.yaml')
    img = video.get_average(100) #### takes average of 100 frames
    video.stop()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ### this is for putting more templates
    if not op.isdir(op.join(dir, 'pytrack_res', 'templates')):
        os.mkdir(op.join(dir, 'pytrack_res', 'templates'))
    if not op.isdir(op.join(dir, 'pytrack_res', 'templates', setup)):
        os.mkdir(op.join(dir, 'pytrack_res', 'templates', setup))
        os.mkdir(op.join(dir, 'pytrack_res', 'templates', setup, 'temp'))
    cv2.imwrite(op.join(dir, 'pytrack_res', 'templates', setup, 'temp', setup+'_'+_timestr+'.png'),img)

    if onlyIm:
        return None

    """
    Get arenas
    """
    thresh = 0.9
    arenas = []
    while len(arenas) != 4:
        img_rgb =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) ### this is for coloring
        ### Template matching function
        loc, vals, w = match_templates(img, 'arena', setup, thresh, dir=dir)
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
    preview(img_rgb, title='Preview arena', topleft='Threshold: {}'.format(thresh), hold=True)



    """
    Get spots
    """
    labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
    geometry = {}
    for ia, arena in enumerate(arenas):
        arena_img = img[arena[1]:arena[1]+w, arena[0]:arena[0]+w]
        c_arena = (arena[0]+w/2, arena[1]+w/2)
        if c_arena[1]<700:
            label='top'
        else:
            label='bottom'
        if c_arena[0]<700:
            label += 'left'
        else:
            label += 'right'
        spots = []
        thresh = 0.99
        min_spots = 6
        max_spots = 8
        while len(spots) < min_spots:
            img_rgb =  cv2.cvtColor(arena_img,cv2.COLOR_GRAY2RGB) ### this is for coloring
            ### Template matching function
            loc, vals, ws = match_templates(arena_img, 'yeast', setup, thresh, dir=dir)
            patches = get_peak_matches(loc, vals, ws, img_rgb, arena=arena)
            spots = patches
            ### Required to have 6 yeast spots detected, lower threshold if not matched
            """
            elif len(spots) > max_spots:
                thresh += 0.001
                thresh = round(thresh,3)
                print('Too many yeast spots detected. Increase matching threshold to {}.'.format(thresh))
            """
            if len(spots) < min_spots:
                thresh -= 0.005
                thresh = round(thresh,3)
                print('Not enough yeast spots detected. Decrease matching threshold to {}.'.format(thresh))
            else:
                #print('Detected 6 yeast spots. Exiting spot detection.')
                spotdf = {'x': [], 'y': [], 'angle': [], 'distance': [], 'orientation': []}
                inner, outer = [], []
                for pt in spots:
                    ept = (int(round(pt[0]+ws/2)), int(round(pt[1]+ws/2)))
                    rx, ry = pt[0]+arena[0]+ws/2, pt[1]+arena[1]+ws/2
                    r = 10 * (w/50.)
                    dist = get_distance([rx, ry], c_arena)
                    angle = get_angle([c_arena[0], -c_arena[1]], [rx, -ry])
                    if angle < 0:
                        angle = 2*np.pi+angle
                    orientation = angle%(np.pi/6)
                    if orientation > np.pi/12:
                        orientation = orientation - np.pi/6
                    if dist > 1.5*r:
                        outer.append((rx, ry))
                    else:
                        inner.append((rx, ry))
                    spotdf['x'].append(rx-c_arena[0])
                    spotdf['y'].append(-(ry-c_arena[1]))
                    spotdf['distance'].append(dist)
                    spotdf['angle'].append(angle)
                    spotdf['orientation'].append(orientation)
                    cv2.circle(img_rgb, ept, int(ws/2), (255,0,255), 1)
                    cv2.circle(img_rgb, ept, 1, (255,0,255), 1)

                inner_est, outer_est = None, None
                if len(inner) == 3:
                    inner_est = (int(round(sum([inn[0] for inn in inner])/len(inner)-arena[0])), int(round(sum([inn[1] for inn in inner])/len(inner)-arena[1])))
                if len(outer) == 3:
                    outer_est = (int(round(sum([out[0] for out in outer])/len(outer)-arena[0])), int(round(sum([out[1] for out in outer])/len(outer)-arena[1])))
                if inner_est is None and outer_est is None:
                    mean_est = (int(round(w/2)), int(round(w/2)))
                elif inner_est is None:
                    mean_est = outer_est
                elif outer_est is None:
                    mean_est = inner_est
                else:
                    mean_est = ( int(round(inner_est[0]+outer_est[0])/2), int(round(inner_est[1]+outer_est[1])/2))
                cv2.circle(img_rgb, mean_est, 1, (0,255,0), 1)

                spotdf = pd.DataFrame(spotdf)
                mean_orient = spotdf['orientation'].mean()
                correct_spots = {'x': [], 'y': [], 's': []}
                for i, angle in enumerate(np.arange(mean_orient+np.pi/3,2*np.pi+mean_orient+np.pi/3, np.pi/3)):
                    for j in range(2):
                        x, y, s = (j+1)*r * np.cos(angle+j*np.pi/6), (j+1) *r*np.sin(angle+j*np.pi/6), i%2
                        correct_spots['x'].append(x)
                        correct_spots['y'].append(y)
                        if s == 0:
                            correct_spots['s'].append('yeast')
                        else:
                            correct_spots['s'].append('sucrose')
                correct_spots = pd.DataFrame(correct_spots)
                all_spots = []
                for index, row in correct_spots.iterrows():
                    if row['s'] == 'yeast':
                        color = (0,165,255)
                    else:
                        color = (255, 144, 30)
                    x, y = row['x']+mean_est[0], -row['y']+mean_est[1]
                    scale = w/50.
                    all_spots.append({'x': row['x']/scale, 'y': row['y']/scale, 'r': 1.5, 'substr': row['s']})
                    cv2.circle(img_rgb, (int(x), int(y)), int(ws/2), color, 1)
                    cv2.circle(img_rgb, (int(x), int(y)), 1, color, 1)
        geometry['fly{:02}'.format(ia+1)] = {   'arena': {'radius': w/2, 'outer': 260.0, 'scale': w/50., 'x': float(mean_est[0]+arena[0]), 'y': float(mean_est[1]+arena[1]), 'name': label}, 'food_spots': all_spots}
        preview(img_rgb, title='Preview spots', topleft='Arena: {}, threshold: {}'.format(label, thresh), hold=True)
    print('save geometry to {}'.format(outfile))
    write_yaml(outfile, geometry)
    return geometry

def manual_geometry(_fullpath, _timestr):
    setup = op.basename(_fullpath).split('_')[0]
    video = VideoCapture(_fullpath, 0)
    dir = op.dirname(_fullpath)
    if not op.isdir(op.join(dir, 'pytrack_res', 'arena')):
        os.mkdir(op.join(dir, 'pytrack_res', 'arena'))
    outfile = op.join(dir, 'pytrack_res','arena', setup+'_arena_' +_timestr+'_manual.yaml')
    img = video.get_average(100) #### takes average of 100 frames
    video.stop()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
    Manual entry
    """
    centers = []
    widths = []
    angles = []
    for fly in range(4):
        print('Fly {}:'.format(fly+1))
        pts = []
        for each in ['first', 'second', 'third']:
            pts.append(input('Type coordinates of {} inner yeast spot: '.format(each)))
        if pts[-1] is None:
            pts = [(0,0), (0,0), (0,0)]
        else:
            pts = [each.split(' ') for each in pts]
            pts = [(int(el[0]), int(el[1])) for el in pts]
        centers.append((sum([el[0] for el in pts])/3, sum([el[1] for el in pts])/3))
        centers[-1] = (int(round(centers[-1][0])), int(round(centers[-1][1])))
        r = get_distance(pts[0],centers[-1])
        widths.append(5*r)
        angles.append(np.arctan2(-(pts[0][1]-centers[-1][1]), pts[0][0]-centers[-1][0]))

    """
    Get arenas
    """
    img_rgb =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) ### this is for coloring
    for pt, w in zip(centers, widths):
        #for ept in pts:
        #    cv2.circle(img_rgb, ept, 1, (255,255,0), 2)
        cv2.circle(img_rgb, pt, int(w/2), (0,255,0), 1)
        cv2.circle(img_rgb, pt, 1, (0,255,0), 2)
    preview(img_rgb, title='Preview arena', topleft='manual', hold=True)

    """
    Get spots
    """
    labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
    geometry = {}
    ia = 0
    for pt, w, a in zip(centers, widths, angles):
        correct_spots = {'x': [], 'y': [], 's': []}
        for i, angle in enumerate(np.arange(a, 2*np.pi+a, np.pi/3)):
            for j in range(2):
                r = w/5.
                x, y, s = (j+1)*r * np.cos(angle+j*np.pi/6), (j+1)*r*np.sin(angle+j*np.pi/6), i%2
                correct_spots['x'].append(x)
                correct_spots['y'].append(y)
                if s == 0:
                    correct_spots['s'].append('yeast')
                else:
                    correct_spots['s'].append('sucrose')
        correct_spots = pd.DataFrame(correct_spots)
        all_spots = []
        for index, row in correct_spots.iterrows():
            if row['s'] == 'yeast':
                color = (0,165,255)
            else:
                color = (255, 144, 30)
            x, y = row['x']+pt[0], -row['y']+pt[1]
            scale = w/50.
            all_spots.append({'x': float(row['x']/scale), 'y': float(row['y']/scale), 'r': 1.5, 'substr': row['s']})
            cv2.circle(img_rgb, (int(round(x)), int(round(y))), int(0.15*r), color, 1)
            cv2.circle(img_rgb, (int(round(x)), int(round(y))), 1, color, 1)
        geometry['fly{:02}'.format(ia+1)] = {   'arena': {'radius': float(w/2), 'outer': 260.0, 'scale': float(w/50.), 'x': float(pt[0]), 'y': float(pt[1]), 'name': labels[ia]}, 'food_spots': all_spots}
        arena_img = img_rgb[pt[1]-int(w/2):pt[1]+int(w/2), pt[0]-int(w/2):pt[0]+int(w/2)]
        preview(arena_img, title='Preview spots', topleft='Arena: {}'.format(labels[ia]), hold=True)
        ia += 1
    print('save geometry to {}'.format(outfile))
    write_yaml(outfile, geometry)
    return geometry
