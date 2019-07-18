# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:02:36 2019

@author: LiXiaoGang

# tf.contrib.layers.variance_scaling_initializer(最适用于relu)
# tf.contrib.layers.xavier_initializer(等价于tf.glorot_uniform_initializer,tf.glorot_normal_initializer最适用于sigmoid,tanh) 
# tf.truncated_normal_initializer(stddev=0.01)
https://blog.csdn.net/qq_27825451/article/details/88707423

"""


import tensorflow as tf

 
def self_attention(Tensor_q,Tensor_k,Tensor_v,d_k):     
    '''
    Scaled Dot-Product Attention
    head = softmax((Q*K^T)/d_k^(1/2))*V
    '''
    Tensor_k_T = tf.transpose(Tensor_k,perm=[1,0],name='transpose')
    scale = tf.div(tf.matmul(Tensor_q,Tensor_k_T),tf.sqrt(tf.cast(d_k,dtype=tf.float32)),name='scale')
    attention = tf.nn.softmax(scale,axis=-1,name='attention')
    head_output = tf.matmul(attention,Tensor_v,name='head_output')
    return head_output


def multi_head_attention(Tensor_q,Tensor_k,Tensor_v,d_k,d_v,d_model,num_head):
    '''
    Multi-Head Attention
    '''
    outputs = []
    weight_init = tf.truncated_normal_initializer(stddev=0.01)
    for i in range(num_head):
        
        # Projections matrices
        w_q = tf.get_variable(name='w_q/'+str(i),shape=[d_model,d_k],initializer=weight_init)    
        w_k = tf.get_variable(name='w_k/'+str(i),shape=[d_model,d_k],initializer=weight_init)    
        w_v = tf.get_variable(name='w_v/'+str(i),shape=[d_model,d_v],initializer=weight_init)
        
        # Linear
        Tensor_q_linear = tf.matmul(Tensor_q,w_q,name='Tensor_q_linear/'+str(i))
        Tensor_k_linear = tf.matmul(Tensor_k,w_k,name='Tensor_k_linear/'+str(i))
        Tensor_v_linear = tf.matmul(Tensor_v,w_v,name='Tensor_v_linear/'+str(i))
        
        # Self attention 
        head_output = self_attention(Tensor_q_linear,Tensor_k_linear,Tensor_v_linear,d_k)
        outputs.append(head_output)
    concat = tf.concat(outputs,axis=1,name='concat')
    
    # Linear
    w_o = tf.get_variable(name='w_o',shape=[num_head*d_v,d_model],initializer=weight_init)
    multi_head_output = tf.matmul(concat,w_o,name='multi_head_output')
    multi_head_output = tf.expand_dims(multi_head_output,0,name='expand_dims')
    return multi_head_output
 

def ffn(multi_head_output,d_ff,d_model):
    '''
    Position-wise Feed-Forward Networks
    '''
    ffn_conv1 = tf.layers.conv1d(multi_head_output,d_ff,1,activation=tf.nn.relu,use_bias=True,name='ffn_conv1')
    ffn_conv2 = tf.layers.conv1d(ffn_conv1,d_model,1,activation=None,use_bias=True,name='ffn_conv2')
    residual_output = tf.add(multi_head_output,ffn_conv2,name='residual_output')
    residual_output = tf.squeeze(residual_output,name='squeeze')
    return residual_output
    
    
def decoder(Tensor_q,Tensor_k,Tensor_v,d_k,d_v,d_model,num_head,d_ff):
    '''
    Decoder
    '''
    multi_head_output = multi_head_attention(Tensor_q,Tensor_k,Tensor_v,d_k,d_v,d_model,num_head)
    decoder_output = ffn(multi_head_output,d_ff,d_model)
    return decoder_output