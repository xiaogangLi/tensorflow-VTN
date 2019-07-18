# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:46:14 2019

@author: LiXiaoGang
https://tensorflow.google.cn/
https://github.com/tensorflow/serving
https://blog.csdn.net/thriving_fcl/article/details/75213361
https://blog.csdn.net/loveliuzz/article/details/81128024
https://www.cnblogs.com/mbcbyq-2137/p/10044837.html
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model

https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

https://www.jianshu.com/p/e5e36ffde809
https://www.jianshu.com/p/9221fbf52c55

batch norm(bug):https://blog.csdn.net/zaf0516/article/details/89958962
请问在将pb文件转为tflite文件时如果模型里有batch normal应该如何操作？
https://www.zhihu.com/question/318251292?sort=created
https://blog.csdn.net/computerme/article/details/80836060
https://stackoverflow.com/questions/45800871/tensorflow-save-restore-batch-norm#
https://github.com/tensorflow/serving/issues/986
          
"""

import os
import sys
import VTN
import shutil
import parameters
import numpy as np
import pandas as pd
import read_data as rd
import tensorflow as tf
import save_inference_model


def load_clip_name(path,status,balance):
    labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))    # load label.txt
    all_clips_name = rd.read_dataset(path,labels,status,seed=66,balance=balance)    # train set
    mean_image = np.load(os.path.join(path,'Data','Train','mean_image.npy'))    # read mean image
    return all_clips_name,mean_image
        
 
def net_placeholder(batch_size=None):
    clip_X = tf.placeholder(tf.float32,shape=[parameters.SEQ_SIZE,
                                              parameters.IN_HEIGHT,
                                              parameters.IN_WIDTH,
                                              parameters.IN_CHANNEL],name='Input')
    clip_Y = tf.placeholder(tf.float32,shape=[batch_size,
                                              parameters.NUM_CLASSESS],name='Label')
#    isTraining = tf.placeholder_with_default(False,shape=None,name='Batch_norm')
    isTraining = tf.placeholder(tf.bool,name='Batch_norm')
    return clip_X,clip_Y,isTraining
    

def net_loss(clip_Y,logits):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=clip_Y,logits=logits),name='loss')
    return loss
    

def val(sess,clip_X,clip_Y,isTraining,Softmax_output,test_batch_size,path,status,balance):
    all_clips_name,mean_image = load_clip_name(path,status,balance)
    acc_count = 0
    
    for j in range(len(all_clips_name)):
        if (j*test_batch_size)>len(all_clips_name):
            break
        Y,X = rd.read_minibatch(j,test_batch_size,all_clips_name,mean_image,status)
        feed_dict = {clip_X:X,clip_Y:Y,isTraining:False}    # Equivalent to {clip_X:X,clip_Y:Y} 
        softmax = sess.run(Softmax_output,feed_dict=feed_dict)

        # Compute clip-level accuracy
        for one_output,one_clip_Y in zip(softmax,Y):
            if np.argmax(one_output) == np.argmax(one_clip_Y):
                acc_count += 1
    accuracy = (acc_count/(len(all_clips_name)*1.0))
    return accuracy


def training_net():
    all_clips_name,mean_image = load_clip_name(parameters.path,'Train',True)
    clip_X,clip_Y,isTraining = net_placeholder(None)
    logits,Softmax_output = VTN.vtn(clip_X,isTraining)
    loss = net_loss(clip_Y,logits)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):        
        train_step = tf.train.AdamOptimizer(parameters.LEARNING_RATE).minimize(loss)
     
    Saver = tf.train.Saver(var_list=tf.global_variables())    # Saver
    with tf.Session() as sess:    # Launch the graph in a session.
        
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.summary.FileWriter(os.path.join(parameters.path,'Model'), sess.graph)     
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        for i in range(parameters.TRAIN_STEPS):
            Y,X = rd.read_minibatch(i,parameters.BATCH_SIZE,all_clips_name,mean_image,'Train')
            feed_dict = {clip_X:X,clip_Y:Y,isTraining:True}
            _,loss_ = sess.run([train_step,loss],feed_dict=feed_dict)
            
            # ------------------------------- Save model --------------------------
            if i % 100 == 0: 
                # accuracy on val clips
                val_acc = val(sess,clip_X,clip_Y,isTraining,Softmax_output,1,parameters.path,'Val',False)
                print('\nVal_accuracy = %g\n' % (val_acc))
                
                # Way 1 : saving checkpoint model
                if sys.argv[1] == 'CHECKPOINT':                    
                    if os.path.exists(parameters.CHECKPOINT_MODEL_SAVE_PATH) and (i==0):
                        shutil.rmtree(parameters.CHECKPOINT_MODEL_SAVE_PATH)
                    Saver.save(sess,os.path.join(parameters.CHECKPOINT_MODEL_SAVE_PATH,parameters.MODEL_NAME+str(i))) 
                        
                #  Way 2 : saving pb model       
                elif sys.argv[1] == 'PB':  
                    if os.path.exists(parameters.PB_MODEL_SAVE_PATH):
                        shutil.rmtree(parameters.PB_MODEL_SAVE_PATH)
                    save_inference_model.save_model(sess,parameters.PB_MODEL_SAVE_PATH,clip_X,Softmax_output,isTraining) 
                else:
                    print('The argument is incorrect for the way saving model!')
                    sys.exit(0)
            print('===>Step %d: loss = %g ' % (i,loss_))
    

def main():
    training_net()
     
if __name__ == '__main__':
    main()