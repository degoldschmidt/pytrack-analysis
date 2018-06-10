import cv2
import numpy as np
import threading
import os
import os.path as op
import platform, subprocess
import imageio
from collections import deque

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
PixelDiff algorithm
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
        print(len(xy[0]))
        for fly, each in enumerate(xy[0]):
            x[:,fly], y[:,fly] = np.array(xy[0][fly])[:nframes], np.array(xy[1][fly])[:nframes]
            tx[:,fly], ty[:,fly] = np.array(txy[0][fly])[:nframes], np.array(txy[1][fly])[:nframes]
        for i in range(nframes-1):
            if i%int(nframes/20)==0:
                print('Run PixelDiff: frames processed: {:3d}%'.format(int(100*i/nframes)))
            if i == 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i+self.sf)
            ret, frame = self.cap.read()
            for fly, each in enumerate(xy[0]):
                if not (np.isnan(x[i,fly]) and np.isnan(y[i,fly])):
                    xi, yi = int(round(x[i,fly])), int(round(y[i,fly]))
                    txi, tyi = int(round(tx[i,fly])), int(round(ty[i,fly]))
                    #print('fly {}: ({}, {}) ({}, {})'.format(fly, xi, yi, txi, tyi))
                    px[i, fly] = frame[yi, xi,0]
                    tpx[i, fly] = frame[tyi, txi,0]
                    cv2.circle(frame, (xi, yi), 3, (255,0,255), 1)
                    cv2.circle(frame, (txi, tyi), 3, (25,255,25), 1)
            if show and i%300==0:
                resized_image = cv2.resize(frame, (500,500), interpolation = cv2.INTER_CUBIC)
                cv2.imshow('Frame', resized_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.stop()
        return px, tpx

def retrack(video, nframes, start_frame=0, background_frames=100, adaptation_rate=0., threshold_value=15, threshold_type=cv2.THRESH_BINARY, subtraction_method='Dark', skip=0, show=False, async=False):
    from pytrack_analysis.geometry import get_distance
    if async:
        precap = VideoCaptureAsync(video, start_frame)
        cap = VideoCaptureAsync(video, start_frame)
    else:
        precap = VideoCapture(video, start_frame)
        cap = VideoCapture(video, start_frame)
    ### data arrays
    xy = np.zeros((nframes, 2, 4))
    hxy = np.zeros((nframes, 2, 4))
    txy = np.zeros((nframes, 2, 4))
    a = np.zeros((nframes, 4))
    xy.fill(np.nan)
    hxy.fill(np.nan)
    txy.fill(np.nan)
    precap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if async:
        precap.start()
    ### set up background
    #for iframe in range(background_frames):
    for ii,iframe in enumerate(np.random.choice(nframes, background_frames)):
        precap.set(cv2.CAP_PROP_POS_FRAMES, start_frame+iframe)
        ret, frame = precap.read()
        if ii == 0:
            ### arrays
            image = np.zeros(frame.shape, dtype=frame.dtype)
            difference = np.zeros(frame.shape, dtype=frame.dtype)
            background = np.zeros(frame.shape, dtype=np.float32)
            output = np.zeros(frame.shape, dtype=frame.dtype)
            outputgray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY).astype(np.uint8)
        background += frame
        if ii == background_frames-1:
            background /= background_frames
    if show:
        resized_image = cv2.resize(background.astype(np.uint8), (700,700), interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Frame', resized_image)
        cv2.waitKey(0)
    precap.stop()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame+skip)
    if async:
        cap.start()
    for iframe in range(nframes):
        if iframe%10000==0:
            print('frame:', iframe)
        ret, frame = cap.read()
        if frame is None:
            break
        ### set up background
        image = frame.copy()
        if subtraction_method == 'Bright':
            difference = image - background
        elif subtraction_method == 'Dark':
            difference = background - image
        else:
            difference = np.abs(image - background)
        if adaptation_rate > 0:
            background = adaptation_rate * image + (1 - adaptation_rate) * background
        ret, output = cv2.threshold(difference, threshold_value, 255, threshold_type)
        outputgray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY).astype(np.uint8)
        im2, contours, hierarchy = cv2.findContours(outputgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


        contours = [cnt for cnt in contours if len(cnt)<100] #len(cnt)>18 and
        if len([cnt for cnt in contours if len(cnt)>18]) > 4:
            contours = [cnt for cnt in contours if len(cnt)<75]
        contours = sorted(contours, key=len, reverse=True)[:4]
        ##print([len(cnt) for cnt in contours])
        mask = np.zeros(frame.shape[:2], np.uint8)
        #cv2.drawContours(mask, contours[:4], -1, (255), -1)
        for fly, cnt in enumerate(contours[:4]):
            if len(cnt) > 5:
                (x,y),(mi,ma), angle = cv2.fitEllipse(cnt)
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(mask, ellipse,(255),-1)
                if x < 700 and y < 700:
                    nfly = 0
                elif x > 700 and y < 700:
                    nfly = 1
                elif x < 700 and y > 700:
                    nfly = 2
                else:
                    nfly = 3
                xy[iframe,0,nfly], xy[iframe,1,nfly] = x, y
                a[iframe, nfly] = len(cnt)
                ### compute head and tail position (NOT validated)
                angle = np.radians(angle)+np.pi/2
                hx, hy = x+0.5*ma*np.cos(angle), y+0.5*ma*np.sin(angle)
                tx, ty = x-0.5*ma*np.cos(angle), y-0.5*ma*np.sin(angle)
                hxy[iframe,0,nfly], hxy[iframe,1,nfly] = hx, hy
                txy[iframe,0,nfly], txy[iframe,1,nfly] = tx, ty
                ### check continuity
                if iframe > 0:
                    ohx, ohy = hxy[iframe-1,0,nfly], hxy[iframe-1,1,nfly]
                    dist_head = get_distance([ohx, ohy], [hx, hy])
                    dist_tail = get_distance([ohx, ohy], [tx, ty])
                    if dist_tail < dist_head:
                        hxy[iframe,0,nfly], hxy[iframe,1,nfly] = tx, ty
                        txy[iframe,0,nfly], txy[iframe,1,nfly] = hx, hy
                ### draw onto frames
                if show:
                    for i in range(100):
                        nx, ny, ox, oy = xy[iframe-i,0,nfly], xy[iframe-i,1,nfly], xy[iframe-i-1,0,nfly], xy[iframe-i-1,1,nfly]
                        #if not np.isnan(nx) and not np.isnan(ox):
                            #cv2.line(frame, (int(nx), int(ny)), (int(ox), int(oy)),(255,0,255),2)
                    #ellipse = cv2.fitEllipse(cnt)
                    #cv2.ellipse(frame,ellipse,(0,255,0),1)
                    #cv2.circle(frame,(int(hxy[iframe,0,nfly]), int(hxy[iframe,1,nfly])),3,[255,255,0],-1)
        if show:
            if iframe%10==0:
                image_res = cv2.bitwise_and(frame, frame, mask = mask)
                mask = cv2.bitwise_not(mask)
                bk = np.full(frame.shape, 255, dtype=np.uint8)  # white bk
                bk_masked = cv2.bitwise_and(bk, bk, mask=mask)
                final = cv2.bitwise_or(image_res, bk_masked)
                newframe = np.zeros((200,200,3), dtype=np.uint8)
                thisx = int(round(xy[iframe,0,0]))
                thisy = int(round(xy[iframe,1,0]))
                thisangle = np.arctan2(txy[iframe,1,0]-hxy[iframe,1,0], txy[iframe,0,0]-hxy[iframe,0,0])
                M = cv2.getRotationMatrix2D((xy[iframe,0,0], xy[iframe,1,0]), -thisangle,1)
                dst = cv2.warpAffine(final,M,(frame.shape[0],frame.shape[1]))
                newframe[:100,:100,:] = dst[thisy-50:thisy+50, thisx-50:thisx+50,:]
                thisx = int(round(xy[iframe,0,1]))
                thisy = int(round(xy[iframe,1,1]))
                thisangle = np.arctan2(txy[iframe,1,1]-hxy[iframe,1,1], txy[iframe,0,1]-hxy[iframe,0,1])
                M = cv2.getRotationMatrix2D((xy[iframe,0,1], xy[iframe,1,1]),np.degrees(-thisangle),1)
                dst = cv2.warpAffine(final,M,(frame.shape[0],frame.shape[1]))
                newframe[100:,:100,:] = dst[thisy-50:thisy+50, thisx-50:thisx+50,:]
                thisx = int(round(xy[iframe,0,2]))
                thisy = int(round(xy[iframe,1,2]))
                thisangle = np.arctan2(txy[iframe,1,2]-hxy[iframe,1,2], txy[iframe,0,2]-hxy[iframe,0,2])
                M = cv2.getRotationMatrix2D((xy[iframe,0,2], xy[iframe,1,2]),np.degrees(-thisangle),1)
                dst = cv2.warpAffine(final,M,(frame.shape[0],frame.shape[1]))
                newframe[:100,100:,:] = dst[thisy-50:thisy+50, thisx-50:thisx+50,:]
                thisx = int(round(xy[iframe,0,3]))
                thisy = int(round(xy[iframe,1,3]))
                thisangle = np.arctan2(txy[iframe,1,3]-hxy[iframe,1,3], txy[iframe,0,3]-hxy[iframe,0,3])
                M = cv2.getRotationMatrix2D((xy[iframe,0,3], xy[iframe,1,3]),np.degrees(-thisangle),1)
                dst = cv2.warpAffine(final,M,(frame.shape[0],frame.shape[1]))
                newframe[100:,100:,:] = dst[thisy-50:thisy+50, thisx-50:thisx+50,:]
                resized_image = cv2.resize(newframe, (700,700), interpolation = cv2.INTER_CUBIC)
                cv2.imshow('Frame', resized_image)
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break
    cap.stop()
    cv2.destroyAllWindows()
    return xy, hxy, txy, a

"""
Retracking videos
"""
class Retracking:
    def __init__(self, video, start_frame=0, background_frames=10, adaptation_rate=0., threshold_value=30, threshold_type=cv2.THRESH_BINARY, subtraction_method='Dark'):
        self.cap = VideoCapture(video, start_frame)
        self.sf = start_frame
        ### setting up
        self.bgfr = background_frames
        self.alpha = adaptation_rate
        self.thr_val = threshold_value
        self.thr_type = threshold_type
        self.sm = subtraction_method
        ### arrays
        ret, frame = self.cap.read()
        self.im = np.zeros(frame.shape, dtype=frame.dtype)
        self.diff = np.zeros(frame.shape, dtype=frame.dtype)
        self.bg = np.zeros(frame.shape, dtype=np.float32)
        self.out = np.zeros(frame.shape, dtype=frame.dtype)
        self.outgray = cv2.cvtColor(self.out,cv2.COLOR_BGR2GRAY).astype(np.uint8)
        self.sf = start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def run(self, nframes, show=False):
        from pytrack_analysis.geometry import get_distance
        ### data arrays
        xy = np.zeros((nframes, 2, 4))
        hxy = np.zeros((nframes, 2, 4))
        txy = np.zeros((nframes, 2, 4))
        ###
        image = self.im
        difference = self.diff
        background = self.bg
        output = self.out
        outputgray = self.outgray
        ### set up background
        for iframe in range(self.bgfr):
            ret, frame = self.cap.read()
            background += frame
            if iframe == self.bgfr-1:
                background /= self.bgfr
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.sf)
        for iframe in range(nframes):
            if iframe%1000==0:
                print('frame:', iframe)
            ret, frame = self.cap.read()
            if frame is None:
                break
            ### set up background
            image = frame.copy()
            if self.sm == 'Bright':
                difference = image - background
            elif self.sm == 'Dark':
                difference = background - image
            else:
                difference = np.abs(image - background)
            if self.alpha > 0:
                background = self.alpha * image + (1 - self.alpha) * background
            ret, output = cv2.threshold(difference, self.thr_val, 255, self.thr_type)
            outputgray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY).astype(np.uint8)
            im2, contours, hierarchy = cv2.findContours(outputgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(output,contours,-1,(0,255,0),1)
            for fly, cnt in enumerate(contours):
                if len(cnt) > 10:
                    (x,y),(mi,ma), angle = cv2.fitEllipse(cnt)
                    if x < 700 and y < 700:
                        nfly = 0
                    elif x > 700 and y < 700:
                        nfly = 1
                    elif x < 700 and y > 700:
                        nfly = 2
                    else:
                        nfly = 3
                    xy[iframe,0,nfly], xy[iframe,1,nfly] = x, y
                    ### compute head and tail position (NOT validated)
                    angle = np.radians(angle)+np.pi/2
                    hx, hy = x+0.5*ma*np.cos(angle), y+0.5*ma*np.sin(angle)
                    tx, ty = x-0.5*ma*np.cos(angle), y-0.5*ma*np.sin(angle)
                    hxy[iframe,0,nfly], hxy[iframe,1,nfly] = hx, hy
                    txy[iframe,0,nfly], txy[iframe,1,nfly] = tx, ty
                    ### check continuity
                    if iframe > 0:
                        ohx, ohy = hxy[iframe-1,0,nfly], hxy[iframe-1,1,nfly]
                        dist_head = get_distance([ohx, ohy], [hx, hy])
                        dist_tail = get_distance([ohx, ohy], [tx, ty])
                        if dist_tail < dist_head:
                            hxy[iframe,0,nfly], hxy[iframe,1,nfly] = tx, ty
                            txy[iframe,0,nfly], txy[iframe,1,nfly] = hx, hy
                    ### draw onto frames
                    if show:
                        for i in range(100):
                            nx, ny, ox, oy = xy[iframe-i,0,nfly], xy[iframe-i,1,nfly], xy[iframe-i-1,0,nfly], xy[iframe-i-1,1,nfly]
                            cv2.line(frame, (int(nx), int(ny)), (int(ox), int(oy)),(255,0,255),2)
                        ellipse = cv2.fitEllipse(cnt)
                        cv2.ellipse(frame,ellipse,(0,255,0),1)
                        cv2.circle(frame,(int(hxy[iframe,0,nfly]), int(hxy[iframe,1,nfly])),3,[255,255,0],-1)
            if show:
                if iframe%1==0:
                    resized_image = cv2.resize(frame, (500,500), interpolation = cv2.INTER_CUBIC)
                    cv2.imshow('Frame', resized_image)
                    k = cv2.waitKey(1) & 0xff
                    if k == 27:
                        break
        self.cap.stop()
        cv2.destroyAllWindows()
        return xy, hxy, txy


"""
Writes overlay
"""
class ShowOverlay:
    def __init__(self, video, start_frame=0):
        self.cap = VideoCapture(video, start_frame)
        #ret, frame = self.cap.read()
        self.sf = start_frame

    def run(self, xy, txy, bxy, pxls, nframes, show=True):
        x, y = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        tx, ty = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        bx, by = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        px, tpx = np.zeros((nframes, len(xy[0]))), np.zeros((nframes, len(xy[0])))
        flipped = np.zeros((nframes, len(xy[0])))
        for fly, each in enumerate(xy[0]):
            x[:,fly], y[:,fly] = np.array(xy[0][fly])[:nframes], np.array(xy[1][fly])[:nframes]
            tx[:,fly], ty[:,fly] = np.array(txy[0][fly])[:nframes], np.array(txy[1][fly])[:nframes]
            bx[:,fly], by[:,fly] = np.array(bxy[0][fly])[:nframes], np.array(bxy[1][fly])[:nframes]
            px[:,fly], tpx[:,fly] = pxls[fly][0][:nframes], pxls[fly][1][:nframes]
        for i in range(nframes-1):
            if i%int(nframes/10)==0:
                print('Run ShowOverlay: frames processed: {:3d}%'.format(int(100*i/nframes)))
            #if i == 0:
            intev = 300
            if i%intev==0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i+self.sf)
                ret, frame = self.cap.read()
                for fly, each in enumerate(xy[0]):
                    if not (np.isnan(x[i,fly]) and np.isnan(y[i,fly])):
                        bxi, byi = int(round(bx[i,fly])), int(round(by[i,fly]))
                        xi, yi = int(round(x[i,fly])), int(round(y[i,fly]))
                        txi, tyi = int(round(tx[i,fly])), int(round(ty[i,fly]))
                        post = i + intev
                        if post > flipped.shape[0]-1:
                            post = flipped.shape[0]-1
                        mpx, mtpx = np.mean(px[i:post,fly]), np.mean(tpx[i:post,fly])
                        cv2.circle(frame, (xi, yi), 3, (255,0,255), 1)
                        cv2.circle(frame, (txi, tyi), 3, (25,255,25), 1)
                        if mpx > mtpx + 5:
                            cv2.circle(frame, (bxi+40, byi+40), 5, (0,0,255), -1)
                            flipped[i:i+intev] = 1
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
                    #resized_image = cv2.resize(resized_image, (500,500), interpolation = cv2.INTER_CUBIC)
                    cv2.imshow('Frame', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        self.cap.stop()
        return flipped

"""
Writes overlay
"""
class WriteOverlay:
    def __init__(self, video, outfolder=None):
        self.cap = VideoCapture(video, 0)
        self.out = os.path.join(outfolder, os.path.basename(video)[:-4])
        if not op.isdir(self.out):
            os.mkdir(self.out)

    def __del__(self):
        self.cap.stop()

    def run(self, xy, hxy, start_frame, end_frame, jumpat, view, fly, bool=[]):
        x, y = np.array(xy[0]), np.array(xy[1])
        hx, hy = np.array(hxy[0]), np.array(hxy[1])
        x0, y0 = int(view[0]), int(view[1])
        w, h = int(view[2]), int(view[3])
        bools = []
        for b in bool:
            bools.append(np.array(b))
        of = os.path.join(self.out, 'fly{}_{:06d}.avi'.format(fly+1, jumpat))
        #print('Save video to ', of)
        self.writer = cv2.VideoWriter(of, cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (w, h))

        nframes = end_frame - start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(nframes-1):
            ret, frame = self.cap.read()
            #if i%int(nframes/10)==0:
            #    print('Run WriteOverlay: frames processed: {:3d}%'.format(int(100*i/nframes)))
            if ret:
                cv2.circle(frame, (int(hx[i]), int(hy[i])), 3, (255,0,255), 1)
                cv2.line(frame, (int(x[i]), int(y[i])), (int(hx[i]), int(hy[i])), (255,0,255), 1)
                resized_image = frame[y0:y0+h, x0:x0+w]
                cv2.putText(resized_image, '{}'.format(i+start_frame), (40, 50), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,0,255),1, cv2.LINE_AA)
                #cv2.putText(resized_image,'{:.3f}'.format(dr[i]), (w-80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,0,255),1, cv2.LINE_AA)
                if bools[1][i] == 0:
                    cv2.circle(resized_image, (w-50, 50), 25, (0,0,255), -1)
                if bools[1][i] == 1:
                    cv2.circle(resized_image, (w-50, 50), 25, (0,255,0), -1)
                if bools[0][i] == 1:
                    cv2.circle(resized_image, (w-50, h-50), 25, (0,0,255), -1)
                try:
                    self.writer.write(resized_image)
                except cv2.error:
                    print('frame: ', i)
                    print('head: ({}, {})'.format(int(hx[i]), int(hy[i])))
                    print('body: ({}, {})'.format(int(x[i]), int(y[i])))
                    print('frame resized to (width: {} - {}, height: {} - {})'.format(x0,x0+w, y0,y0+h))
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
        toff = 1000
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
