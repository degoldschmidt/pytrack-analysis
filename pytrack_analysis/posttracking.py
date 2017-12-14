import os
import numpy as np
from pytrack_analysis.cli import colorprint

"""
Returns dictionary of all raw data files
"""
def get_files(raw, session, video_folder):
    session_folder = os.path.join(raw, "{:02d}".format(session))
    if os.path.isdir(session_folder):
        print("\nStart post-tracking analysis for video session: {:02d}".format(session))
        dtstamp, timestampstr = get_time(session_folder)
        file_dict = {
                        "data" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "fly" in eachfile and timestampstr in eachfile],
                        "food" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "food" in eachfile and timestampstr in eachfile],
                        "geometry" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "geometry" in eachfile and timestampstr in eachfile][0],
                        "timestart" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "timestart" in eachfile and timestampstr in eachfile][0],
                        "video" : [os.path.join(video_folder, eachfile) for eachfile in os.listdir(video_folder) if timestampstr in eachfile][0],
                    }
        print("[post_tracking]\tTimestamp:", dtstamp.strftime("%A, %d. %B %Y %H:%M"))
        return file_dict, dtstamp, timestampstr
    else:
        print("Session {:02d} not found.".format(session))

"""
Returns number of frame skips, big frame skips (more than 10 frames) and maximum skipped time between frames
"""
def get_frame_skips(datal, skips='frame_dt', println=False, printmore=False):
    frameskips = datal[0][skips]
    max_skip = np.amax(frameskips)
    max_skip_arg = frameskips.idxmax()
    for odata in datal:
        oframeskips = odata[skips]
        if np.any(oframeskips != frameskips):
            colorprint('[post_tracking]\tWARNING: not same frameskips', color='warning')
    total = frameskips.shape[0]
    strict_skips = np.sum(frameskips > (1/30)+(1/30))
    easy_skips = np.sum(frameskips > (1/30)+(1/3))
    if println:
        print('[post_tracking]\tdetected frameskips: {:} ({:3.3f}% of all frames)'.format(strict_skips, 100*strict_skips/total))
    if printmore:
        print('[post_tracking]\tskips of more than 1 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/30), strict_skips, 100*strict_skips/total))
        print('[post_tracking]\tskips of more than 10 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/3), easy_skips, 100*easy_skips/total))
    return strict_skips, easy_skips, max_skip, max_skip_arg

"""
Returns session numbers as list based on arguments
"""
def get_session_list(N, *args):
    start = 1
    end = N
    nott = []
    if len(args) > 0:
        if type(args[0]) is str:
            start = int(args[0]) + 1
    if len(args) > 1:
        if type(args[1]) is str:
            end = int(args[1]) + 1
    if len(args) > 2:
        if type(args[2]) is str:
            nott = [int(each) for each in args[2].split(',')]
    outlist = [i for i in range(start,end)]
    for each in nott:
        try:
            outlist.remove(each)
        except ValueError:
            pass
    if len(args) > 3:
        if type(args[3]) is str:
            outlist = [int(each) for each in args[3].split(',')]
    return outlist

"""
Returns timestamp from session folder
"""
def get_time(session):
    from datetime import datetime
    for each in os.listdir(session):
        if "timestart" in each:
            any_file = each
    timestampstr = any_file.split('.')[0][-19:]
    dtstamp = datetime.strptime(timestampstr, "%Y-%m-%dT%H_%M_%S")
    return dtstamp, timestampstr[:-3]
