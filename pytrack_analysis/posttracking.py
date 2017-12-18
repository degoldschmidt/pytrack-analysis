import os
import numpy as np
from pytrack_analysis.cli import colorprint, flprint, prn

def mistracks(data):
    pass


def frameskips(data, dt=None):
    if dt is None:
        data.skips = get_frameskips(data.raw_data, println=True)
    else:
        data.skips = get_frameskips(data.raw_data, dt=dt, println=True)


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
