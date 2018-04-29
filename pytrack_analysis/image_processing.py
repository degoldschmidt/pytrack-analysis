import cv2
import numpy as np
import threading
import os
import os.path as op
import platform, subprocess
import imageio

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
Writes overlay
"""
class WriteOverlay:
    def __init__(self, video, start_frame=0, view=None, outfile=None):
        self.cap = VideoCapture(video, start_frame)
        out = os.path.join(os.path.dirname(video), 'videos', outfile)
        self.view = view
        self.writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(view[2]), int(view[3])))

    def run(self, xy, state, nframes):
        x, y = np.array(xy[0]), np.array(xy[1])
        x0, y0 = int(self.view[0]), int(self.view[1])
        w, h = int(self.view[2]), int(self.view[3])
        for i in range(nframes-1):
            ret, frame = self.cap.read()
            if i%1800==0:   ### every minute
                print('frame: {}'.format(i))
            if ret:
                cv2.circle(frame, (int(x[i]), int(y[i])), 2, (255,0,255), 1)
                if state[i]:
                    cv2.circle(frame, (x0+10, y0+10), 10, (0,0,255), -1)
                resized_image = frame[y0:y0+h, x0:x0+w]
                self.writer.write(resized_image)
        self.cap.stop()
        self.writer.release()


"""
Detect jumps and mistracking
"""
class JumpDetection:
    def __init__(self, video, start_frame=0):
        self.cap = VideoCapture(video, start_frame)
        #self.cap.start()
        self.ret, self.frame = self.cap.grabbed, self.cap.frame

    def displacements(self, x, y, dt):
        dx = np.append(0, np.diff(x))
        dy = np.append(0, np.diff(y))
        dr = np.sqrt(dx*dx + dy*dy)
        return np.divide(dr,dt)

    def run(self, data, nframes):
        x, y = np.array(data['Item1.Item1.X']), np.array(data['Item1.Item1.Y'])
        for i in range(nframes-1):
            if i%1800==0:   ### every minute
                print('frame: {} {}'.format(i, (int(x[i]), int(y[i]))))
                if self.ret == True:
                    cv2.circle(self.cap.frame, (int(x[i]), int(y[i])), 5, (0,165,255), 1)
                    resized_image = self.cap.frame[int(y[i])-50:int(y[i])+50, int(x[i])-50:int(x[i])+50]#cv2.resize(self.cap.frame, (350, 350))
                    cv2.imshow('Frame',resized_image)
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            self.ret, self.frame = self.cap.read()
        self.cap.stop()

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

def get_peak_matches(loc, vals, w, img_rgb, arena=None, show_all=False, show_peaks=False):
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

def main():
    import pandas as pd
    from pytrack_analysis.yamlio import read_yaml
    video = "/media/degoldschmidt/DATA_BACKUP/data/tracking/videos/cam01_2017-11-24T11_42_04.avi"
    data = "/home/degoldschmidt/post_tracking/cam01_fly01_2017-11-24T11_42_04.csv"
    data2 = "/home/degoldschmidt/post_tracking/DIFF_013.csv"
    meta = "/home/degoldschmidt/post_tracking/DIFF_013.yaml"

    meta_dict = read_yaml(meta)
    sf = meta_dict['video']['first_frame']
    nframes = meta_dict['video']['nframes']
    df = pd.read_csv(data, sep="\s+").loc[sf:,:]

    jd = JumpDetection(video, start_frame=sf)
    df['displacements'] = jd.displacements(df['Item1.Item1.X'], df['Item1.Item1.Y'], df['Item4'])
    print(df['displacements'])
    #jd.run(df, nframes)

if __name__ == '__main__':
    from pytrack_analysis import Multibench
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
