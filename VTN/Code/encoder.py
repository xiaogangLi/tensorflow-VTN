# -*- coding: utf-8 -*-


import parameters
import tensorflow as tf


def mobilenetv2(clip_X,mode):
    '''
    Implementation of Mobilenet V2.
    Architecture: https://arxiv.org/abs/1801.04381
    '''
    
    width_scale=1
    input_channel = 32
    
              # t, c,  n, s
    arguments=[[1, 16, 1, 1],
               [6, 24, 2, 2],
               [6, 32, 3, 2],
               [6, 64, 4, 2],
               [6, 96, 3, 1],
               [6, 160, 3, 2],
               [6, 320, 1, 1]]
     
    # building first layer
    with tf.variable_scope('Conv2d3x3',reuse=None):
        input_channel = int(input_channel*width_scale)
        
        conv_3x3 = tf.layers.conv2d(clip_X,
                                    input_channel,(3,3),
                                    strides=(2,2),
                                    padding='same',
                                    activation=None,
                                    use_bias=False,
                                    name='conv')
        conv_bn_3x3 = tf.layers.batch_normalization(conv_3x3,training=mode,name='conv_bn_3x3')
        activation = tf.nn.relu(conv_bn_3x3,name='relu')
    
    
    # building inverted residual blocks
    m = 0
    with tf.variable_scope('Bottlenecks',reuse=None):
        for t, c, n, s in arguments:
            m += 1
            
            with tf.variable_scope('Blocks'+str(m),reuse=None):
                output_channel = int(c * width_scale)
                
                for i in range(n):
                    with tf.variable_scope('block'+str(i),reuse=None):                    
                        filters = t * input_channel
                        
                        if i==0: 
                            # pointwise layer
                            pw = tf.layers.conv2d(activation,
                                                  filters,(1,1),
                                                  strides=(1,1),
                                                  padding='valid',
                                                  activation=None,
                                                  use_bias=False,
                                                  name='conv_non-linear')
                            pw_bn = tf.layers.batch_normalization(pw,training=mode,name='pw_bn')
                            pw_relu = tf.nn.relu(pw_bn,name='pw_relu')
                            
                            # depthwise layer
                            dw = tf.contrib.layers.separable_conv2d(pw_relu,num_outputs=None,
                                                                    kernel_size=[3,3],
                                                                    depth_multiplier=1,
                                                                    stride=[s,s],
                                                                    padding='SAME',
                                                                    activation_fn=None,
                                                                    biases_initializer=None)  
                            dw_bn = tf.layers.batch_normalization(dw,training=mode,name='dw_bn')
                            dw_relu = tf.nn.relu(dw_bn,name='dw_relu')
                            
                            # pointwise linear layer
                            plw = tf.layers.conv2d(dw_relu,
                                                   output_channel,(1,1),
                                                   strides=(1,1),
                                                   padding='valid',
                                                   activation=None,
                                                   use_bias=False,
                                                   name='conv_linear')
                            pwl_bn = tf.layers.batch_normalization(plw,training=mode,name='pwl_bn')
                            
                            # residual connection
                            if (s == 1) and (filters == output_channel):
                                activation = pwl_bn + activation
                            else:
                                activation = pwl_bn
                        else:
                            # pointwise layer
                            pw = tf.layers.conv2d(activation,
                                                  filters,(1,1),
                                                  strides=(1,1),
                                                  padding='valid',
                                                  activation=None,
                                                  use_bias=False,
                                                  name='conv_non-linear')
                            pw_bn = tf.layers.batch_normalization(pw,training=mode,name='pw_bn')
                            pw_relu = tf.nn.relu(pw_bn,name='pw_relu')
                            
                            # depthwise layer
                            dw = tf.contrib.layers.separable_conv2d(pw_relu,num_outputs=None,
                                                                    kernel_size=[3,3],
                                                                    depth_multiplier=1,
                                                                    stride=[1,1],
                                                                    padding='SAME',
                                                                    activation_fn=None,
                                                                    biases_initializer=None)
                            dw_bn = tf.layers.batch_normalization(dw,training=mode,name='dw_bn')
                            dw_relu = tf.nn.relu(dw_bn,name='dw_relu')
                            
                            # pointwise linear layer
                            plw = tf.layers.conv2d(dw_relu,
                                                   output_channel,(1,1),
                                                   strides=(1,1),
                                                   padding='valid',
                                                   activation=None,
                                                   use_bias=False,
                                                   name='conv_linear')
                            pwl_bn = tf.layers.batch_normalization(plw,training=mode,name='pwl_bn')
                            
                            # residual connection
                            if (s == 1) and (filters == output_channel):
                                activation = pwl_bn + activation
                            else:
                                activation = pwl_bn
                        input_channel = output_channel
    
         
    # building last several layers
    with tf.variable_scope('Conv2d1x1',reuse=None):
        conv_1x1 = tf.layers.conv2d(activation,
                                    parameters.d_model,(1,1),
                                    strides=(1,1),
                                    padding='valid',
                                    activation=None,
                                    use_bias=False,
                                    name='conv')
        conv_bn_1x1 = tf.layers.batch_normalization(conv_1x1,training=mode,name='conv_bn_1x1')
        activation = tf.nn.relu(conv_bn_1x1,name='relu')
        
        shape = activation.get_shape().as_list()
        width = shape[1]
        height = shape[2]
        output = tf.layers.average_pooling2d(activation,
                                             pool_size=(height,width),
                                             strides=(1,1),
                                             padding='valid',
                                             name='glo_avg_epool2d')
        output = tf.squeeze(output,name='squeeze')  
    return output           
