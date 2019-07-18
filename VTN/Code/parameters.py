# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:55:44 2019

@author: LiXiaoGang
"""

import os
import pandas as pd


d_k = 64
d_v = 64
d_ff = 2048
d_model = 512
num_head = 8
num_stacks = 1
encoder = 'mobilenetv2'    # Supported encoders include mobilenetv1,mobilenetv2,mobilenetv3

LEARNING_RATE = 0.0001
TRAIN_STEPS = 1000
BATCH_SIZE = 1    # 只能等于1

SEQ_SIZE = 16    # clip 的深度，也即是clip视频片段的帧数

IN_DEPTH = 2*SEQ_SIZE       
IN_HEIGHT = 224   # 帧的高度
IN_WIDTH = 224    # 帧的宽度
IN_CHANNEL = 3    # 3通道的RGB图像
STRIDE = 32     # 滑动窗取clips时的步长，clip_depth <= stride 表示取clip时，clip之间不存在重叠

rate = 0.2   # 验证集和测试集在整个数据集（Raw_Data）中的比例都为0.2
remove_mean_image = True

path = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))

NUM_CLASSESS = len(labels.Class_name)    # 视频类别数量
MODEL_NAME = 'model.ckpt-'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(path,'Model','checkpoint')
PB_MODEL_SAVE_PATH = os.path.join(path,'Model','pb')