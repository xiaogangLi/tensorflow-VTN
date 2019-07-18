# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:45:21 2019
https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md
@author: LiXiaoGang
"""

import tensorflow as tf


SIGNATURE_KEY = 'SIGNATURE'
INPUT_KEY = 'INPUT'
OUTPUT_KEY = 'OUTPUT'
BATCH_NORM_KEY = 'BATCH_NORM'
MODEL_NAME = 'VTN'

def save_model(sess,model_save_path,input_tensor,output_tensor,isTraining_tensor):
    
    model_input = {INPUT_KEY:tf.saved_model.utils.build_tensor_info(input_tensor),
                   BATCH_NORM_KEY:tf.saved_model.utils.build_tensor_info(isTraining_tensor)}
    model_output = {OUTPUT_KEY:tf.saved_model.utils.build_tensor_info(output_tensor)}
    
    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=model_input, outputs=model_output,method_name=SIGNATURE_KEY)
    
    builder = tf.saved_model.builder.SavedModelBuilder(model_save_path)
    builder.add_meta_graph_and_variables(sess,[MODEL_NAME], signature_def_map={SIGNATURE_KEY:signature})
    builder.save()


def inference(model_save_path,clip):
    
    with tf.Session() as sess:
        
        meta_graph_def = tf.saved_model.loader.load(sess,[MODEL_NAME],model_save_path)
        signature = meta_graph_def.signature_def    # 从meta_graph_def中取出SignatureDef对象
        
        # 从signature中找出具体输入输出的tensor name
        bn_tensor_name = signature[SIGNATURE_KEY].inputs[BATCH_NORM_KEY].name
        x_tensor_name = signature[SIGNATURE_KEY].inputs[INPUT_KEY].name
        y_tensor_name = signature[SIGNATURE_KEY].outputs[OUTPUT_KEY].name
                                 
        
        # 获取tensor
        bn_tensor = sess.graph.get_tensor_by_name(bn_tensor_name)
        x_tensor = sess.graph.get_tensor_by_name(x_tensor_name)
        y_tensor = sess.graph.get_tensor_by_name(y_tensor_name)
        
        # inference
        output = sess.run(y_tensor, feed_dict={x_tensor:clip,bn_tensor:False})
        return output