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
        self.cap = cv2.VideoCapture(self.src,0)
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
class PixelDiff:
    def __init__(self, video, start_frame=0):
        self.cap = VideoCapture(video, start_frame)
        #ret, frame = self.cap.read()
        self.sf = start_frame

    def run(self, xy, txy, nframes, show=True):
        x, y = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        tx, ty = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        px, tpx = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        for fly, each in enumerate(xy[0]):
            x[:,fly], y[:,fly] = np.array(xy[0][fly])[:nframes], np.array(xy[1][fly])[:nframes]
            tx[:,fly], ty[:,fly] = np.array(txy[0][fly])[:nframes], np.array(txy[1][fly])[:nframes]
        for i in range(nframes-1):
            if i%int(nframes/20)==0:
                print('frames processed: {:3d}%'.format(int(100*i/nframes)))
            if i == 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i+self.sf)
            ret, frame = self.cap.read()
            for fly, each in enumerate(xy[0]):
                if not (np.isnan(x[i,fly]) and np.isnan(y[i,fly])):
                    xi, yi = int(round(x[i,fly])), int(round(y[i,fly]))
                    txi, tyi = int(round(tx[i,fly])), int(round(ty[i,fly]))
                    #print('fly {}: ({}, {}) ({}, {})'.format(fly, xi, yi, txi, tyi))
                    px[i, fly] = np.mean(frame[yi-1:yi+2, xi-1:xi+2,0])
                    tpx[i, fly] = np.mean(frame[tyi-1:tyi+2, txi-1:txi+2,0])
                    cv2.circle(frame, (xi, yi), 3, (255,0,255), 1)
                    cv2.circle(frame, (txi, tyi), 3, (25,255,25), 1)
            if show and i%300==0:
                resized_image = cv2.resize(frame, (500,500), interpolation = cv2.INTER_CUBIC)
                cv2.imshow('Frame', resized_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.stop()
        return px, tpx

"""
Writes overlay
"""
class ShowOverlay:
    def __init__(self, video, start_frame=0):
        self.cap = VideoCapture(video, start_frame)
        #ret, frame = self.cap.read()
        self.sf = start_frame

    def run(self, xy, txy, bxy, nframes, show=True):
        x, y = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        tx, ty = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        bx, by = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        px, tpx = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        for fly, each in enumerate(xy[0]):
            x[:,fly], y[:,fly] = np.array(xy[0][fly])[:nframes], np.array(xy[1][fly])[:nframes]
            tx[:,fly], ty[:,fly] = np.array(txy[0][fly])[:nframes], np.array(txy[1][fly])[:nframes]
            bx[:,fly], by[:,fly] = np.array(bxy[0][fly])[:nframes], np.array(bxy[1][fly])[:nframes]
        for i in range(nframes-1):
            if i%int(nframes/20)==0:
                print('frames processed: {:3d}%'.format(int(100*i/nframes)))
            #if i == 0:
            if i%30==0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i+self.sf)
                ret, frame = self.cap.read()
                for fly, each in enumerate(xy[0]):
                    if not (np.isnan(x[i,fly]) and np.isnan(y[i,fly])):
                        bxi, byi = int(round(bx[i,fly])), int(round(by[i,fly]))
                        xi, yi = int(round(x[i,fly])), int(round(y[i,fly]))
                        txi, tyi = int(round(tx[i,fly])), int(round(ty[i,fly]))
                        #print('fly {}: ({}, {}) ({}, {})'.format(fly, xi, yi, txi, tyi))
                        px[i, fly] = np.mean(frame[yi-1:yi+2, xi-1:xi+2,0])
                        tpx[i, fly] = np.mean(frame[tyi-1:tyi+2, txi-1:txi+2,0])
                        cv2.circle(frame, (xi, yi), 3, (255,0,255), 1)
                        cv2.circle(frame, (txi, tyi), 3, (25,255,25), 1)
                if show:
                    resized_image = frame.copy()
                    resized_image = resized_image[:200, :200]
                    bxi, byi = int(round(bx[i,0])), int(round(by[i,0]))
                    resized_image[:100, :100] =  frame[byi-50:byi+50, bxi-50:bxi+50]
                    bxi, byi = int(round(bx[i,1])), int(round(by[i,1]))
                    resized_image[:100, 100:] =  frame[byi-50:byi+50, bxi-50:bxi+50]
                    bxi, byi = int(round(bx[i,2])), int(round(by[i,2]))
                    resized_image[100:, :100] =  frame[byi-50:byi+50, bxi-50:bxi+50]
                    bxi, byi = int(round(bx[i,3])), int(round(by[i,3]))
                    resized_image[100:, 100:] =  frame[byi-50:byi+50, bxi-50:bxi+50]
                    cv2.imshow('Frame', resized_image)
                    if cv2.waitKey(1000) & 0xFF == ord('q'):
                        break
        self.cap.stop()
        return px, tpx


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
def match_templates(img, object_name, setup, threshold, method=cv2.TM_CCOEFF_NORMED, dir=None):
    idir = op.join(dir, 'pytrack_res', 'templates')
    files = [op.join(idir, setup, _file) for _file in os.listdir(os.path.join(idir, setup)) if object_name in _file and not _file.startswith('.')]
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

def preview(img, title='preview geometry', topleft='', hold=False):
    preview = cv2.resize(img, (700, 700))
    font = cv2.FONT_HERSHEY_SIMPLEX
    if hold:
        toff = 0
    else:
        toff = 1
    cv2.putText(preview, topleft, (10, 30), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(title+' (press any key to continue)',preview)
    if platform.system() == 'Darwin':
        tmpl = 'tell application "System Events" to set frontmost of every process whose unix id is {} to true'
        script = tmpl.format(os.getpid())
        output = subprocess.check_call(['/usr/bin/osascript', '-e', script])
    cv2.waitKey(toff)
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
