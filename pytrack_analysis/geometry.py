import numpy as np
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
