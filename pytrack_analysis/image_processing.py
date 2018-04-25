import cv2
import numpy as np
import threading
import os
import os.path as op
import platform, subprocess

class VideoCapture:
    def __init__(self, src, var):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, var)
        self.grabbed, self.frame = self.cap.read()

    def get_average(self, var):
        avg = np.float32(self.frame)
        for i in range(100):
            ret, frame = self.read()
            cv2.accumulateWeighted(frame,avg,0.01)
            img = cv2.convertScaleAbs(avg)
        return img

    def get(self, var):
        self.cap.get(var)

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def set_frame(self, var):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, var)

    def read(self):
        self.grabbed, self.frame = self.cap.read()
        frame = self.frame.copy()
        grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.cap.release()

class VideoCaptureAsync:
    def __init__(self, src, var):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, var)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def get(self, var):
        self.cap.get(var)

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def set_frame(self, var):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, var)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

"""
Returns locations that match templates
"""
def match_templates(img, object_name, setup, threshold, method=cv2.TM_CCOEFF_NORMED):
    files = [op.join('..', 'media', 'templates', setup, _file) for _file in os.listdir(os.path.join('..', 'media', 'templates', setup)) if object_name in _file]
    templates = [cv2.imread(_file,0) for _file in files]
    size = templates[0].shape[0]    ### templates should have same size
    result = [cv2.matchTemplate(img,template,method) for template in templates]
    loc = None
    vals = None
    for r in result:
        if loc is None:
            loc = list(np.where( r >= threshold ))
            vals = r[np.where( r >= threshold )]
        else:
            temp = list(np.where( r >= threshold ))
            tempvals = r[np.where( r >= threshold )]
            loc[0] = np.append(loc[0], temp[0])
            loc[1] = np.append(loc[1], temp[1])
            vals = np.append(vals, tempvals)
    return loc, vals, size

def get_peak_matches(loc, vals, w, img_rgb, show_all=False, show_peaks=False):
    patches = []
    maxv = []
    for i, pt in enumerate(zip(*loc[::-1])):
        v = vals[i]
        if show_all:
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + w), (0,0,255), 2)
        if len(patches) == 0:
            patches.append(pt)
            maxv.append(v)
        else:
            flagged = False
            outside = len(patches)*[False]
            for j, each_patch in enumerate(patches):
                if abs(each_patch[0]-pt[0]) < w and abs(each_patch[1]-pt[1]) < w:
                    if v > maxv[j]:
                        patches[j] = pt
                        maxv[j] = v
                        break
                elif abs(each_patch[0]-pt[0]) < w and abs(each_patch[1]-pt[1]) < w:
                    flagged = True
                elif abs(each_patch[0]-pt[0]) > w or abs(each_patch[1]-pt[1]) > w:
                    outside[j] = True
            if all(outside):
                patches.append(pt)
                maxv.append(v)
    if show_peaks:
        for pt in patches:
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + w), (0,0,255), 1)
        #print('found {} patches.'.format(len(patches)))
    return patches

def preview(img, title='preview geometry', topleft=''):
    preview = cv2.resize(img, (700, 700))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(preview, topleft, (10, 30), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(title+' (press any key to continue)',preview)
    if platform.system() == 'Darwin':
        tmpl = 'tell application "System Events" to set frontmost of every process whose unix id is {} to true'
        script = tmpl.format(os.getpid())
        output = subprocess.check_call(['/usr/bin/osascript', '-e', script])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
