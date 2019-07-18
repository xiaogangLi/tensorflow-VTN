# -*- coding: utf-8 -*-


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
    labels = pd.read_csv(os.path.join(path,'Label_Map','label.txt'))
    all_clips_name = rd.read_dataset(path,labels,status,seed=66,balance=balance)
    mean_image = np.load(os.path.join(path,'Data','Train','mean_image.npy'))
    return all_clips_name,mean_image
        
 
def net_placeholder(batch_size=None):
    clip_X = tf.placeholder(tf.float32,shape=[parameters.SEQ_SIZE,
                                              parameters.IN_HEIGHT,
                                              parameters.IN_WIDTH,
                                              parameters.IN_CHANNEL],name='Input')
    clip_Y = tf.placeholder(tf.float32,shape=[batch_size,
                                              parameters.NUM_CLASSESS],name='Label')
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
        feed_dict = {clip_X:X,clip_Y:Y,isTraining:False}
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
     
    Saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter(os.path.join(parameters.path,'Model'), sess.graph)     
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        for i in range(parameters.TRAIN_STEPS):
            Y,X = rd.read_minibatch(i,parameters.BATCH_SIZE,all_clips_name,mean_image,'Train')
            feed_dict = {clip_X:X,clip_Y:Y,isTraining:True}
            _,loss_ = sess.run([train_step,loss],feed_dict=feed_dict)
            
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
