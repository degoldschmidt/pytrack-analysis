import imageio
import os
import numpy as np
import warnings
from pytrack_analysis.cli import colorprint, flprint, prn

special = [u"\u2196", u"\u2197", u"\u2199", u"\u2198"]

"""
Returns number of frame skips, big frame skips (more than 10 frames) and maximum skipped time between frames
"""
def get_frameskips(datal, dt='frame_dt', println=False, printmore=False):
    frameskips = np.array(datal[0].loc[:,dt])
    max_skip = np.amax(frameskips)
    max_skip_arg = frameskips.argmax()
    for odata in datal:
        oframeskips = np.array(odata.loc[:,dt])
        if np.any(oframeskips != frameskips):
            prn(__name__)
            colorprint('WARNING: not same frameskips', color='warning')
    total = frameskips.shape[0]
    strict_skips = np.sum(frameskips > (1/30)+(1/30))
    easy_skips = np.sum(frameskips > (1/30)+(1/3))
    if println:
        if 100*strict_skips/total < 0.1:
            prn(__name__)
            print('detected frameskips: {:} ({:3.3f}% of all frames)'.format(strict_skips, 100*strict_skips/total))
        else:
            prn(__name__)
            flprint('detected frameskips: ')
            colorprint('{:} ({:3.3f}% of all frames)'.format(strict_skips, 100*strict_skips/total))
    if printmore:
        prn(__name__)
        print('skips of more than 1 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/30), strict_skips, 100*strict_skips/total))
        prn(__name__)
        print('skips of more than 10 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/3), easy_skips, 100*easy_skips/total))
    return {"Strict frameskips": strict_skips,"Long frameskips": easy_skips, "Max frameskip duration":  max_skip, "Max frameskip index": max_skip_arg}


def frameskips(data, dt=None):
    if dt is None:
        data.skips = get_frameskips(data.raw_data, println=True)
    else:
        data.skips = get_frameskips(data.raw_data, dt=dt, println=True)

def get_displacements(data, x=None, y=None):
    dx = np.append(0,np.diff(data[x]))
    dy = np.append(0,np.diff(data[y]))
    displ = np.sqrt(dx**2 + dy**2)
    return displ

def get_head_tail(data, x=None, y=None, angle=None, major=None):
    head_x = np.array(data[x] + 0.5*data[major]*np.cos(data[angle]))
    head_y = np.array(data[y] + 0.5*data[major]*np.sin(data[angle]))
    tail_x = np.array(data[x] - 0.5*data[major]*np.cos(data[angle]))
    tail_y = np.array(data[y] - 0.5*data[major]*np.sin(data[angle]))
    return {'head_x': head_x, 'head_y': head_y, 'tail_x': tail_x, 'tail_y': tail_y}

def mistracks(data, ix, dr=None, major=None, thresholds=(4, 5)):
    # get displacements
    displ = np.array(data.loc[:,dr])
    displ[np.isnan(displ)] = 0
    # get major axis length
    maj = data[major]
    # two thresholds
    threshold = thresholds[0]
    speed_threshold = thresholds[1]
    # bitwise or
    mask = (maj>threshold) | (displ>speed_threshold)
    # get mistracked frames
    mistracks = data.index[mask]
    # output to console
    prn(__name__)
    flprint('Arena ', special[ix], ' - mistracked frames: ')
    if len(mistracks)<300:
        print(len(mistracks))
    else:
        colorprint(str(len(mistracks)), color='warning')
    # mistracked framed get NaNs
    data.loc[mistracks, ['body_x', 'body_y', 'angle', 'major', 'minor', 'displacement']] = np.nan
    return data

def get_patch_average(x, y, radius=1, image=None):
    pxls = []
    if image is not None:
        for dx in range(-radius, radius+1):
            yr = radius-abs(dx)
            for dy in range(-yr, yr+1):
                pxls.append(image[int(y)+dy, int(x)+dx, 0])
    return np.mean(np.array(pxls))

def get_pixel_flip(data, hx=None, hy=None, tx=None, ty=None, video=None, start=None):
    warnings.filterwarnings("ignore")
    head_x, head_y = np.array(data[hx]), np.array(data[hy])
    tail_x, tail_y = np.array(data[tx]), np.array(data[ty])
    vid = imageio.get_reader(video)
    skip=1
    headpx = np.zeros(head_x.shape)
    tailpx = np.zeros(tail_x.shape)
    for t in range(start, start+head_x.shape[0], skip):
        ### load image
        this_frame = vid.get_data(t)
        i = t-start
        if not (np.isnan(head_x[i]) and np.isnan(head_y[i])):
            headpx[i:i+skip] = get_patch_average(head_x[i], head_y[i], image=this_frame)
        if not (np.isnan(tail_x[i]) and np.isnan(tail_y[i])):
            tailpx[i:i+skip] = get_patch_average(tail_x[i], tail_y[i], image=this_frame)
        if (t-start)%10000==0:
            print(t, headpx[i:i+skip], tailpx[i:i+skip])
    pixeldiff = tailpx - headpx
    warnings.filterwarnings("default")
    vid.close()
    return np.array(pixeldiff<0), headpx, tailpx
