import cv2
import math
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
vidcap = cv2.VideoCapture("Top.mp4")
path = 'C:/Users/User/Desktop/hcl/Hackathon/Images/'
frameRate = vidcap.get(5)
while(vidcap.isOpened()):
    frameId = vidcap.get(1)
    flag,frame = vidcap.read()
    if(flag!=True):
        break
    if(frameId % math.floor(frameRate) == 0):
        filename = path + "/img" + str(int(frameId)) + ".jpg"
        cv2.imwrite(filename,frame)     
vidcap.release()
