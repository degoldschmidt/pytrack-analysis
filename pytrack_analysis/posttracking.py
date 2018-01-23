import imageio
import os
import numpy as np
import pandas as pd
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

def get_displacements(data, x=None, y=None, angle=None):
    dx = np.append(0,np.diff(data[x]))
    dy = np.append(0,np.diff(data[y]))
    displ = np.sqrt(dx**2 + dy**2)
    dar = np.append(0,np.diff(displ))
    ori = np.arctan2(dy,dx)
    diff = np.cos(np.array(data[angle]) - ori)
    for i in range(diff.shape[0]):
        if i > 0:
            if displ[i] < 2.:
                diff[i] = 0.0
        else:
            diff[i] = 0.0
    return displ, dx, dy, ori, diff, dar

def get_head_tail(data, x=None, y=None, angle=None, major=None):
    head_x = np.array(data[x] + 0.5*data[major]*np.cos(data[angle]))
    head_y = np.array(data[y] + 0.5*data[major]*np.sin(data[angle]))
    tail_x = np.array(data[x] - 0.5*data[major]*np.cos(data[angle]))
    tail_y = np.array(data[y] - 0.5*data[major]*np.sin(data[angle]))
    return head_x, head_y, tail_x, tail_y

def mistracks(data, ix, dr=None, major=None, thresholds=(4*8.543, 5*8.543), keep=False):
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
    if not keep:
        print("Do not keep...")
        data.loc[mistracks, ['body_x', 'body_y', 'angle', 'major', 'minor', 'displacement']] = np.nan
    return data

def get_patch_average(x, y, radius=1, image=None):
    pxls = []
    if image is not None:
        for dx in range(-radius, radius+1):
            yr = radius-abs(dx)
            for dy in range(-yr, yr+1):
                #print((x,y), int(y)+dy, int(x)+dx, image[int(y)+dy, int(x)+dx, 0])
                pxls.append(image[int(y)+dy, int(x)+dx, 0])
    return np.mean(np.array(pxls))

def get_pixel_flip(datal, hx=None, hy=None, tx=None, ty=None, offset=None, video=None, start=None):
    warnings.filterwarnings("ignore")
    vid = imageio.get_reader(video)
    skip=1
    heads = []
    tails = []
    headpxs = []
    tailpxs = []
    flips = []
    for data in datal:
        heads.append(np.array(data[[hx, hy]]))
        tails.append(np.array(data[[tx, ty]]))
        headpxs.append(np.zeros(heads[-1].shape[0]))
        tailpxs.append(np.zeros(tails[-1].shape[0]))
    for t in range(start, start+heads[-1].shape[0], skip):
        ### load image
        i = t-start
        this_frame = vid.get_data(t)
        for idx, data in enumerate(datal):
            if not np.any(np.isnan(heads[idx][i,:])):
                headpxs[idx][i:i+skip] = get_patch_average(heads[idx][i,0], heads[idx][i,1], image=this_frame)
            if not np.any(np.isnan(tails[idx][i,:])):
                tailpxs[idx][i:i+skip] = get_patch_average(tails[idx][i,0], tails[idx][i,1], image=this_frame)
        if (t-start)%10000==0:
            print(t, headpxs[0][i:i+skip], tailpxs[0][i:i+skip], headpxs[1][i:i+skip], tailpxs[1][i:i+skip], headpxs[2][i:i+skip], tailpxs[2][i:i+skip], headpxs[3][i:i+skip], tailpxs[3][i:i+skip])
    for data in datal:
        pixeldiff = (tailpxs[idx] - headpxs[idx])
        pixeldiff = np.array(pixeldiff < 0) ## if head brighter than tail -> flip
        flips.append(pixeldiff)
    warnings.filterwarnings("default")
    vid.close()
    return flips, headpxs, tailpxs


def get_corrected_flips(df, _VERBOSE=False):
    hpx, tpx = np.array(df['headpx']), np.array(df['tailpx'])
    pxdf = tpx - hpx
    df['flip'] = np.array(pxdf < 0)
    df['jump'] = df['acc'] > 5.
    df['flipped'] = df['displacement'] < 0.
    jumptimes = np.append(df.index[0], df.query('jump == True').index, df.index[-1])
    if _VERBOSE:
        print(jumptimes)
    for i, each in enumerate(jumptimes[1:]):
        dt = min(jumptimes[i+1]-jumptimes[i], 250)
        mean_flip = np.mean(df.loc[jumptimes[i]:jumptimes[i]+dt, 'flip'])
        mean_align = np.mean(df.loc[jumptimes[i]:jumptimes[i]+dt, 'align'])
        flip_decision = (mean_flip > 0.5 and mean_align < 0.) or mean_flip > 0.9 or mean_align < -0.1
        if _VERBOSE:
            print(jumptimes[i], dt, mean_flip, mean_align, flip_decision)
        if flip_decision:
            nheadx, nheady = np.array(df.loc[jumptimes[i]:jumptimes[i+1], 'tail_x']), np.array(df.loc[jumptimes[i]:jumptimes[i+1], 'tail_y'])
            ntailx, ntaily = np.array(df.loc[jumptimes[i]:jumptimes[i+1], 'head_x']), np.array(df.loc[jumptimes[i]:jumptimes[i+1], 'head_y'])
            df.loc[jumptimes[i]:jumptimes[i+1], 'head_x'] = nheadx
            df.loc[jumptimes[i]:jumptimes[i+1], 'tail_x'] = ntailx
            df.loc[jumptimes[i]:jumptimes[i+1], 'head_y'] = nheady
            df.loc[jumptimes[i]:jumptimes[i+1], 'tail_y'] = ntaily
            df.loc[jumptimes[i]:jumptimes[i+1], 'angle'] -= np.pi
            df.loc[jumptimes[i]:jumptimes[i+1],'flipped'] = True
