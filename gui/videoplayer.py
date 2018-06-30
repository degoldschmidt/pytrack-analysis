import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os.path as op


VIDEODIR = '/media/degoldschmidt/DATA_DENNIS_002/working_data/0007_KPEG'
VIDEOFILE = op.join(VIDEODIR, 'cam01_2018-04-18T15_39_08.avi')
START_FRAME = 4000

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Videoplayer")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)
w = tk.Scale(sliderFrame, from_=0, to=108000, width =10, length=700, resolution=1, orient=tk.HORIZONTAL)
w.pack(expand=True, fill=tk.BOTH)
var = w.get()

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(VIDEOFILE)

def get_background(nframes, bg_frames, startat=0):
    for ii,iframe in enumerate(np.random.choice(nframes, bg_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, startat+iframe)
        ret, frame = cap.read()
        if ii == 0:
            ### arrays
            image = np.zeros(frame.shape, dtype=frame.dtype)
            difference = np.zeros(frame.shape, dtype=frame.dtype)
            background = np.zeros(frame.shape, dtype=np.float32)
            output = np.zeros(frame.shape, dtype=frame.dtype)
            outputgray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY).astype(np.uint8)
        background += frame
        if ii == bg_frames-1:
            background /= bg_frames
    return background

bg = get_background(108000, 100, startat=START_FRAME)

def show_frame():
    var = -1
    if var != w.get():
        var = w.get()
        cap.set(cv2.CAP_PROP_POS_FRAMES, var)
        _, frame = cap.read()
        diff = bg - frame
        ret, mask = cv2.threshold(cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)
        #output = 255 - output
        #output = cv2.bitwise_and(frame, frame, mask = mask)
        outputgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.uint8)
        preview = cv2.flip(outputgray, 1)
        preview = cv2.resize(preview, (700, 700))
        cv2image = preview#cv2.cvtColor(preview, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    lmain.after(100, show_frame)

show_frame()  #Display 2
window.mainloop()  #Starts GUI
