import cv2
vidcap = cv2.VideoCapture('E:\\Dennis\\Google Drive\\PhD Project\\Experiments\\001-DifferentialDeprivation\\data\\videos\\cam01_2017-11-24T08_26_19.avi')
success,image = vidcap.read()
count = 0
success = True
success,image = vidcap.read()
cv2.imwrite("frame.png", image)     # save frame as JPEG file

import pygame as pg
import numpy as np
import pygame.gfxdraw as gfx
import argparse

parser = argparse.ArgumentParser(description='Get file name.')
parser.add_argument('filename', nargs='?', help='file name for analysis')

args = parser.parse_args()
print(args.filename)

###### Game defs
pg.init()
if args.filename == None:
    background = pg.image.load('frame.png')
else:
    background = pg.image.load(args.filename)
backgroundRect = background.get_rect()
size = (width, height) = background.get_size()
print(width, height)
screen = pg.display.set_mode(size)
pg.mouse.set_visible(False)

class Colors:
    black   = ( 0, 0, 0 )
    white   = ( 255, 255, 255 )
    occdark = ( 20, 20, 20 )
    occhell = ( 235, 235, 235 )
    start   = ( 71, 45, 135 )          #472D87
    goal    = (	19, 178, 51 )          #13B233
    hit     = ( 5, 214, 5 )
    missed  = ( 214, 5, 5)

class Cursor:
    def __init__(self, pos, size):
        assert len(pos)==2, "Argument pos is not twodimensional."
        self.x = int(pos[0])
        self.y = int(pos[1])
        self.size = int(size)
        self.sizeb = int(size)
        self.colour = Colors.goal
        self.thickness = 1
        self.blurred = False
        self.typen = 'c'

    def display(self, screen):
        if self.typen == 'c':
            gfx.aacircle(screen, self.x, self.y, self.size, self.colour)
            gfx.aacircle(screen, self.x, self.y, 2, (0,0,0))
        elif self.typen == 'r':
            gfx.rectangle(screen, pg.Rect(self.x-self.size/2, self.y-self.sizeb/2, self.size, self.sizeb), self.colour)
            gfx.aacircle(screen, self.x, self.y, 2, (0,0,0))

    def moveto(self, pos):
        assert len(pos)==2, "Argument pos is not twodimensional."
        self.x = int(pos[0])
        self.y = int(pos[1])

    def setSquare(self):
        self.sizeb = self.size

    def setBigger(self):
        self.size += 1

    def setBBigger(self):
        self.size += 10

    def setSmaller(self):
        self.size -= 1

    def setSSmaller(self):
        self.size -= 10

    def setTaller(self):
        self.sizeb += 1

    def setTTaller(self):
        self.sizeb += 10

    def setShorter(self):
        self.sizeb -= 1

    def setSShorter(self):
        self.sizeb -= 10

mycursor = Cursor((200,400),20)

locked = []
Xvals = []
Yvals = []
Sizes = []
Sizesb = []
Types = []
Names = []

def lock(cursor):
    dcursor = cursor
    dcursor.thickness = 4
    dcursor.colour = (255, 0, 0)
    locked.append(dcursor)
    print('Cursor locked ->')
    print(dcursor.size)
    print('@', (dcursor.x, dcursor.y))
    Xvals.append(dcursor.x)
    Yvals.append(dcursor.y)
    Sizes.append(dcursor.size)
    Sizesb.append(dcursor.sizeb)
    Types.append(dcursor.typen)
    name = input('Name object: ')
    Names.append(name)

done = False
while not done:
    for event in pg.event.get():
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 4: mycursor.setBBigger()
            if event.button == 5: mycursor.setSSmaller()

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE: done = True
            elif event.key == pg.K_UP: mycursor.setBigger()
            elif event.key == pg.K_DOWN: mycursor.setSmaller()
            elif event.key == pg.K_RIGHT: mycursor.setBBigger()
            elif event.key == pg.K_LEFT: mycursor.setSSmaller()
            elif event.key == pg.K_r:
                if mycursor.typen == 'c':
                    mycursor.size *= 2
                mycursor.typen = 'r'
                mycursor.setSquare()
            elif event.key == pg.K_c:
                if mycursor.typen == 'r':
                    mycursor.size = int(mycursor.size/2)
                mycursor.typen = 'c'
            elif event.key == pg.K_q: mycursor.setSquare()
            elif event.key == pg.K_w: mycursor.setTaller()
            elif event.key == pg.K_s: mycursor.setShorter()
            elif event.key == pg.K_d: mycursor.setTTaller()
            elif event.key == pg.K_a: mycursor.setSShorter()
            elif event.key == pg.K_RETURN:
                lock(mycursor)
                mycursor = Cursor((200,400),20)
        elif pg.mouse.get_pressed()[0]:
            lock(mycursor)
            mycursor = Cursor((200,400),20)
        elif event.type == pg.QUIT:
            done = True



    #### FROM HERE ON IS DRAWN ON SCREEN
    screen.fill((0,0,0))
    screen.blit(background.convert_alpha(), backgroundRect)

    mousexy = pg.mouse.get_pos()
    mycursor.moveto(mousexy)
    mycursor.display(screen)
    for i in locked:
        i.display(screen)

    pg.display.flip()

Xvals = np.array(Xvals)
Yvals = np.array(Yvals)
Sizes = np.array(Sizes)
Sizesb = np.array(Sizesb)
Types = np.array(Types)
Names = np.array(Names)

mydt = np.dtype([ ('name', np.str_, 16),
                 ('id', np.str_, 1),
                 ('x', np.int),
                 ('y', np.int),
                 ('size', np.int),
                 ('secsize', np.int) ] )
data = np.zeros(Names.size, dtype=mydt)
data['name'] = Names
data['id'] = Types
data['x'] = Xvals
data['y'] = Yvals
data['size'] = Sizes
data['secsize'] = Sizesb
print(data)
outfile = "../data/processed/arena.cfg"
np.savetxt(outfile, data, delimiter=' ', newline='\n', fmt='%16s %1s %u %u %u %u', header='#Name #Type #X[px] #Y[px] #Size[px] #SizeB[px]')
