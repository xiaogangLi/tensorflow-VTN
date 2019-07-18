# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd


path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

def onehotcode_all_classses(data):
    
    '''
    Function: encode each class using one hot encode style.
    Args:
        data: a DataFrame containing all class names.
    Returns:
        label: a dictionary.
    '''
    
    num_classess = len(data.Class_name)
    oneHotEncode = np.zeros(shape=[num_classess,num_classess])
    
    oneHotEncodeDict = {}
    Class_name = data.Class_name
    
    for i in range(num_classess):
        oneHotEncode[i][i] = 1.0
        oneHotEncodeDict[Class_name[i]] = oneHotEncode[i][:] 
    return oneHotEncodeDict


def onehotencode(video_name_list):
    
    '''
    Function: encode each class using one hot encode style.
    Args:
        video_name_list: a list of class names,e.g,video_name_list = ['Normal_0','Violent_0'] 
    Returns:
        label: a array
    '''
    
    label = []
    oneHotEncodeDict = onehotcode_all_classses(labels)
    
    for i in range(len(video_name_list)):
        label_name = video_name_list[i].split('_')[0]
        label.append(oneHotEncodeDict[label_name])
    label = np.array(label,dtype=np.float32)
    return label


def onehotdecode(one_hot_code):
    
    '''
    Function: decode one hot code as classess.
    Args:
        one_hot_code: a list or an array , the summation of its all elements is 1.0.
    Returns:
        class_name: a class name 
    '''
    
    one_hot_code = list(one_hot_code)
    max_value_index = one_hot_code.index(max(one_hot_code))
    oneHotEncodeDict = onehotcode_all_classses(labels)
    
    for class_name,code in oneHotEncodeDict.items():
        max_idx = np.argmax(code)
        if max_value_index==max_idx:return class_name   
