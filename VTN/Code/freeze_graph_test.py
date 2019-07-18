# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:19:09 2019

@author: LiXiaoGang

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
使用freeze_graph将SavedModel生成的变量固化到pb文件中
"""


import os
import parameters
from tensorflow.python.tools import freeze_graph


# Freeze graph
freeze_graph.freeze_graph(input_graph=os.path.join(parameters.PB_MODEL_SAVE_PATH,'saved_model.pb'),
                          input_saver='',
                          input_binary=False,
                          input_checkpoint=os.path.join(parameters.PB_MODEL_SAVE_PATH,'variables','variables'),
                          output_node_names='softmax_output',
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph=os.path.join(parameters.PB_MODEL_SAVE_PATH,'frozen_graph.pb '),
                          clear_devices=True,
                          initializer_nodes='')