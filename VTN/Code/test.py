# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:06:48 2019

@author: LiXiaoGang
"""

import os
import sys
import parameters
import numpy as np
import encode_label
import pandas as pd
import read_data as rd
import save_inference_model


def read_test_data(path):
    num_clips = int(sys.argv[1])    # 0 < num_clips <= the number of clips in test set.
    labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))    # load label.txt
    all_clips_name = rd.read_dataset(path,labels,'Test',seed=66,balance=False)    # test set
    mean_image = np.load(os.path.join(path,'Data','Train','mean_image.npy'))    # load mean image
    clip_Y,clip_X  = rd.read_minibatch(0,num_clips,all_clips_name,mean_image,'Test')
    return clip_Y,clip_X


def predict(clip_Y,clip_X):
    acc_count = 0
    prediction = []
    for i in range(clip_X.shape[0]):
        one_clip = clip_X[i,:,:,:,:]
        output = save_inference_model.inference(parameters.PB_MODEL_SAVE_PATH,one_clip)    # the output of softmax in model
        if np.argmax(output) == np.argmax(clip_Y[i]):
            acc_count += 1

        # Compute predicted class name
        pred_name = encode_label.onehotdecode(output[0])
        true_name = encode_label.onehotdecode(clip_Y[i])
        prediction.append({'Output':list(output),'Predicted_class_name':pred_name,'True_calss_name':true_name})
    
    # Compute clip-level accuracy
    accuracy = (acc_count/(1.0*clip_X.shape[0]))
    return prediction,accuracy
               

def test_net():
    
    clip_Y,clip_X = read_test_data(parameters.path)
    prediction,accuracy = predict(clip_Y,clip_X)
    print('Clip_accuracy: %g' % accuracy)
    print(prediction)
    return prediction,accuracy
    
    
def main():
    return test_net()
    
    
if __name__ == '__main__':
    prediction,accuracy = main()    