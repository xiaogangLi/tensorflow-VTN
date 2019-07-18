# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:28:00 2019

@author: LiXiaoGang
"""

import sys
import encoder
import decoder
import parameters
import tensorflow as tf


def vtn(clip_X,mode):
    
    # Encode
    with tf.variable_scope('encoder',reuse=None):
        if parameters.encoder == 'mobilenetv2':
            output = encoder.mobilenetv2(clip_X,mode)
 
        elif parameters.encoder == 'mobilenetv3':
            output = encoder.mobilenetv3(clip_X)
        else:
            print('\nThe %s encoder does not exists!\n' % (parameters.encoder))
            sys.exit(0)
    
    # Decode
    with tf.variable_scope('decoder',reuse=None):
        for i in range(parameters.num_stacks):
            with tf.variable_scope('decoder'+str(i),reuse=None):
                output = decoder.decoder(output,output,output,
                                         parameters.d_k,
                                         parameters.d_v,
                                         parameters.d_model,
                                         parameters.num_head,
                                         parameters.d_ff)
                
        clip_logits = tf.reduce_mean(output,axis=1,name='clip_logits')
        clip_logits = tf.expand_dims(clip_logits,0,name='expand_dims')

    # Classifer
    with tf.variable_scope('classifier',reuse=None):
        logits = tf.layers.dense(clip_logits,parameters.NUM_CLASSESS,activation=None,use_bias=True,name='logits')
        softmax_output = tf.nn.softmax(logits,name='softmax_output')
    return logits,softmax_output