# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:46:30 2019

@author: LiXiaoGang
"""

import os 
import random
import cv2 as cv
import numpy as np
import parameters
import encode_label


def read_dataset(path,labels,status,seed=0,balance=True):
    
    all_label_list = []
    Class_name = labels.Class_name
    num_classess = len(labels.Class_name)
    
    max_len = 0
    for i in range(num_classess):        
        if status == 'Train':
            label_list_per_class = os.listdir(os.path.join(path,'Data','Train',Class_name[i]))
        elif status == 'Val':
            label_list_per_class = os.listdir(os.path.join(path,'Data','Val',Class_name[i]))
        elif status == 'Test':
            label_list_per_class = os.listdir(os.path.join(path,'Data','Test',Class_name[i]))
        all_label_list.append(label_list_per_class)
        
        if len(label_list_per_class)> max_len:
            max_len = len(label_list_per_class)
    
    shuffle_all_label_list =[]
    
    # deal with class imbalance by copying samples.
    if  balance == True: 
        for i in range(len(all_label_list)):
            num = int(np.ceil(max_len/len(all_label_list[i])))
            
            z = []
            for j in range(num):
                z = z + all_label_list[i]
            shuffle_all_label_list = shuffle_all_label_list + z[0:max_len]
            
    # do not deal with class imbalance   
    if  balance == False:
        for i in range(len(all_label_list)):
            shuffle_all_label_list = shuffle_all_label_list + all_label_list[i]
        
    # shuffle list 
    random.seed(seed)
    random.shuffle(shuffle_all_label_list)
    return shuffle_all_label_list


def read_minibatch(i,batch_size,all_clips_name,mean_image,status):

    start = (i*batch_size) % len(all_clips_name)
    end = min(start+batch_size,len(all_clips_name))
    
    batch_clips_name = all_clips_name[start:end]
    clip_Y = encode_label.onehotencode(batch_clips_name)
    clip_X = np.zeros([parameters.SEQ_SIZE,
                       parameters.IN_HEIGHT,
                       parameters.IN_WIDTH,
                       parameters.IN_CHANNEL],dtype=np.float32)
    
    if status == 'Test':
        clip_Test = np.zeros([end-start,
                       parameters.SEQ_SIZE,
                       parameters.IN_HEIGHT,
                       parameters.IN_WIDTH,
                       parameters.IN_CHANNEL],dtype=np.float32)
    
    for i in range(min(batch_size,end-start)):
        folder = batch_clips_name[i].split('_')[0]

        if status == 'Train':
            cap = cv.VideoCapture(os.path.join(os.path.dirname(os.getcwd()),'Data','Train',folder,batch_clips_name[i]))
        elif status == 'Val':
            cap = cv.VideoCapture(os.path.join(os.path.dirname(os.getcwd()),'Data','Val',folder,batch_clips_name[i]))
        elif status == 'Test':
            cap = cv.VideoCapture(os.path.join(os.path.dirname(os.getcwd()),'Data','Test',folder,batch_clips_name[i]))
            
        for j in range(parameters.SEQ_SIZE):
            ret,frame = cap.read()
            frame = cv.resize(frame,(parameters.IN_HEIGHT,parameters.IN_WIDTH)).astype(np.float32)
            if parameters.remove_mean_image:
                frame = frame - mean_image    # remove mean image
            clip_X[j,:,:,:] = frame
                  
        if status == 'Test':
            clip_Test[i,:,:,:,:] = clip_X
                     
    if status == 'Test':
        return clip_Y,clip_Test
    else:
         return clip_Y,clip_X