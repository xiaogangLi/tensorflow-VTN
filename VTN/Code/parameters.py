# -*- coding: utf-8 -*-


import os
import pandas as pd


d_k = 64
d_v = 64
d_ff = 2048
d_model = 512
num_head = 8
num_stacks = 4
encoder = 'mobilenetv2'

LEARNING_RATE = 0.0001
TRAIN_STEPS = 1000
BATCH_SIZE = 1    # must be 1

SEQ_SIZE = 16
STRIDE = 2*SEQ_SIZE  
IN_DEPTH = 2*SEQ_SIZE       
IN_HEIGHT = 224
IN_WIDTH = 224
IN_CHANNEL = 3

rate = 0.2    # the proportion of training and validation videos in the raw videos respectively.
remove_mean_image = False

path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

NUM_CLASSESS = len(labels.Class_name)
MODEL_NAME = 'model.ckpt-'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(path,'Model','checkpoint')
PB_MODEL_SAVE_PATH = os.path.join(path,'Model','pb')
