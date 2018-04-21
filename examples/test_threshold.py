import cv2
import numpy as np
from matplotlib import pyplot as plt
import platform
import os

if platform.system() == 'Darwin':
    _dir = '/Users/degoldschmidt/Desktop/tracking_test_data'
else:
    _dir = '/media/degoldschmidt/DATA_BACKUP/data/tracking/sample'
video_file = "cam01_2017-11-24T08_26_19.avi"


cap = cv2.VideoCapture(os.path.join(_dir, video_file)) ### VideoCaptureAsync(os.path.join(self.dir, self.video_file))###imageio.get_reader(os.path.join(self.dir, self.video_file))
#self.video.set(cv2.CAP_PROP_POS_FRAMES, 0); # Where frame_no is the frame you want
yeasts = [_file for _file in os.listdir(_dir) if 'yeast' in _file]
template = cv2.imread('templates/yeast.png',0)
template_arena = cv2.imread('templates/arena.png',0)
w, h = template.shape[::-1]
w2, h2 = template_arena.shape[::-1]
ret, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
res_arena = cv2.matchTemplate(img,template_arena,cv2.TM_CCOEFF_NORMED)
cv2.imwrite('res.png',img)

img_rgb =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

threshold = 0.9
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
loc = np.where( res_arena >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w2, pt[1] + h2), (0,255,0), 2)
cv2.imwrite('res_matched.png',img_rgb)
