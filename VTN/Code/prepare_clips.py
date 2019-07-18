# -*- coding: utf-8 -*-


import os
import random
import cv2 as cv
import parameters
import pandas as pd


path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

for j in range(len(labels.Class_name)):
    i = 0
    class_name = labels.Class_name[j]
    
    src_video_path = os.path.join(path,'Raw_Data',class_name)
    video_names = os.listdir(src_video_path)
    random.shuffle(video_names)
    
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    
    num_train = int(len(video_names)*(1-2*parameters.rate))
    num_val = int(len(video_names)*parameters.rate)
    num_test = int(len(video_names)*parameters.rate) 
    n = 1 
                         
    for name in video_names:
        print('Preprocessing:',name)
        cap = cv.VideoCapture(os.path.join(src_video_path,name))
        
        if (n<=num_train):
            dst_clips_path = os.path.join(path,'Data','Train',class_name)
            
        elif (num_train<n<=(num_train+num_val)):
            dst_clips_path = os.path.join(path,'Data','Val',class_name)
            
        elif n>(num_train+num_val):
            dst_clips_path = os.path.join(path,'Data','Test',class_name)
        n = n + 1
        
        if cap.isOpened():
            num_frames = int(cap.get(7))
            if num_frames < parameters.IN_DEPTH:continue
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
                        
            frame_list = []
            for j in range(num_frames):
                ret, frame = cap.read()
                if ret == True:frame_list.append(frame)
            if len(frame_list) < parameters.IN_DEPTH:continue
            
            for j in range(int(len(frame_list)/parameters.IN_DEPTH)+1):
                
                start = j*parameters.STRIDE
                end = j*parameters.STRIDE + parameters.IN_DEPTH
                
                if (start>len(frame_list)) or (end > len(frame_list)):
                    clips = frame_list[-parameters.IN_DEPTH::]
                else:
                    clips = frame_list[j*parameters.STRIDE:j*parameters.STRIDE+parameters.IN_DEPTH]
    
                i += 1
                out = cv.VideoWriter(os.path.join(dst_clips_path,class_name+'_'+str(i)+'.avi'),fourcc, parameters.IN_DEPTH, (frame_width,frame_height))
                for k in range(parameters.IN_DEPTH):
                    # sampling every second frame from the current video containing 32 frames
                    if (k%2)==0:
                        out.write(clips[k])                       
out.release()                       
cap.release()    
