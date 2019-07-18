# -*- coding: utf-8 -*-
"""
Compute a mean image from the train set.
"""


import os 
import cv2 as cv
import numpy as np
import pandas as pd
import parameters


path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

Class_name = labels.Class_name
num_classess = len(labels.Class_name)

count = 0
mean_image = np.zeros([parameters.IN_HEIGHT,parameters.IN_WIDTH,parameters.IN_CHANNEL],dtype=np.float32)
for i in range(num_classess):
    
    clips_name = os.listdir(os.path.join(path,'Data','Train',Class_name[i]))
    for clip in clips_name:
        cap = cv.VideoCapture(os.path.join(path,'Data','Train',Class_name[i],clip))
        num_frames = int(cap.get(7))
        
        for j in range(num_frames):
            ret,frame = cap.read()
            if ret == True:
                count = count + 1
                frame = cv.resize(frame,(parameters.IN_HEIGHT,parameters.IN_WIDTH)).astype(np.float32)
                mean_image = mean_image + frame
               
mean_image = mean_image/count
np.save(os.path.join(path,'Data','Train','mean_image.npy'),mean_image)
cv.imwrite(os.path.join(path,'Data','Train','mean_image.jpg'),mean_image)
cap.release()    
