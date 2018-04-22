import cv2
import numpy as np
from matplotlib import pyplot as plt
import platform
import os, subprocess

if platform.system() == 'Darwin':
    _dir = '/Users/degoldschmidt/Desktop/tracking_test_data/new_data'
else:
    _dir = '/media/degoldschmidt/DATA_BACKUP/data/tracking/sample'
video_file = "cam02_2018-04-14T16_12_09.avi"#"cam01_2017-11-24T08_26_19.avi"#"cam01_2018-04-14T16_10_40.avi"#"cam01_2017-11-24T08_26_19.avi"

def detect_geometry(_fullpath):
    cap = cv2.VideoCapture(os.path.join(_dir, video_file)) ### VideoCaptureAsync(os.path.join(self.dir, self.video_file))###imageio.get_reader(os.path.join(self.dir, self.video_file))
    setup = video_file.split('_')[0]
    ret, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('res.png',img)
    yeasts = [_file for _file in os.listdir(os.path.join('templates', setup)) if 'yeast' in _file]
    arenas = [_file for _file in os.listdir(os.path.join('templates', setup)) if 'arena' in _file]
    print(yeasts, arenas)
    templates = [cv2.imread(os.path.join('templates', setup, _file),0) for _file in yeasts]
    templates_arena = [cv2.imread(os.path.join('templates', setup, _file),0) for _file in arenas]
    w, h = templates[0].shape[::-1]
    w2, h2 = templates_arena[0].shape[::-1]
    res = [cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) for template in templates]
    res_arena = [cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED) for template in templates_arena]

    img_rgb =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    """
    Get arenas
    """
    arena_threshold = 0.9
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
        if len(patches) == 0:
            patches.append([pt])
        else:
            for i, each_patch in enumerate(patches):
                for eachpt in each_patch:
                    if abs(eachpt[0]-pt[0]) < w2 and abs(eachpt[1]-pt[1]) < h2:
                        patches[i].append(pt)
                        break
            if all([pt not in each_patch for each_patch in patches]):
                patches.append([pt])
    arenas = []
    for each_patch in patches:
        tis = np.array(each_patch)
        arenas.append(np.mean(tis, axis=0))
    for pt in arenas:
        ept = (int(round(pt[0]+w2/2)), int(round(pt[1]+w2/2)))
        cv2.circle(img_rgb, ept, int(w2/2), (0,255,0), 2)
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
        inarena = [(abs(arena[0]+w2/2-tismean[0]-w/2) < w2/2 and abs(arena[1]+w2/2-tismean[1]-w/2) < h2/2) for arena in arenas]
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
