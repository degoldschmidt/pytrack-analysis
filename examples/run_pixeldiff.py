import imageio
from datetime import datetime
import os
import pims
import cv2
import argparse
import platform

import threading

class VideoCaptureAsync:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

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

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

if __name__ == '__main__':
    startTime = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', action="store", dest="method")
    method = parser.parse_args().method

    if platform.system() == 'Darwin':
        folder = "/Users/degoldschmidt/Desktop/tracking_test_data"
    else:
        folder = "/media/degoldschmidt/DATA_BACKUP/data/tracking/videos" #filedialog.askopenfilename(defaultextension='avi')
    video_file = "cam01_2017-11-24T08_26_19.avi"
    n_frames = 1800
    print('Seconds to process: {} s.'.format(n_frames/30))
    if method == 'io':
        video = imageio.get_reader(os.path.join(folder, video_file))
        for i in range(n_frames):
            if i%300==0:
                print('{} s processed.'.format(i/30))
            image = video.get_data(i)
    elif method == 'pims':
        video = pims.Video(os.path.join(folder, video_file))
        for i in range(n_frames):
            if i%300==0:
                print('{} s processed.'.format(i/30))
            image = video[i]
    elif method == 'cv':
        cap = cv2.VideoCapture(os.path.join(folder, video_file))
        for i in range(n_frames):
            if i%300==0:
                print('{} s processed.'.format(i/30))
            _, frame = cap.read()
    elif method == 'cvasync':
        cap = VideoCaptureAsync(os.path.join(folder, video_file))
        cap.set(cv2.CAP_PROP_POS_FRAMES,6101)
        cap.start()
        for i in range(n_frames):
            if i%300==0:
                print('{} s processed.'.format(i/30))
            ret, frame = cap.read()
            if ret == True:
                resized_image = cv2.resize(frame, (350, 350))
                cv2.imshow('Frame',resized_image)
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
        cap.stop()
    print("{} secs.".format(datetime.now() - startTime))
