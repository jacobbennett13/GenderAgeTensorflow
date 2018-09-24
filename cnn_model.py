#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Model.py

Define the TensorFlow model and input/output functions depending on mode

Jacob Bennett, 12/2/17

'''

# Import future Python features while keeping the version and dependancies
# the same
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


# Import libraries we will use
import tensorflow as tf

import gabor as gb

import time


# Log tensorflow information to the terminal window
tf.logging.set_verbosity(tf.logging.INFO)


def normalize(image):
    
    # Normalized image = (I - Imin) / (Imax - Imin)
    I_norm = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    
    return I_norm


# Function to define how to read data into the TensorFlow model
def read_and_decode(filename_queue):
  
    with tf.name_scope('Decode_Queue'):
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized = reader.read(filename_queue)
        
        feature = {'image_raw': tf.FixedLenFeature([], tf.string),
                   'gender_label': tf.FixedLenFeature([], tf.int64),
                   'age_label': tf.FixedLenFeature([], tf.int64)}

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized, features=feature)
        
        # Convert the image data from string back to the numbers
        image_ = tf.decode_raw(features['image_raw'], tf.float32) 
        
        # Normalize image
        image = normalize(image_)
        
        # Cast label data into int32
        age_label = tf.cast(features['age_label'], tf.int64)
        gender_label = tf.cast(features['gender_label'], tf.int64)
        
        # Reshape image data into the original shape
        image.set_shape([128*128])
        
        return image, age_label, gender_label
    
    

def input_fn(filename, batch_size=128):
    
    with tf.name_scope('Batch_Queue'):
        filename_queue = tf.train.string_input_producer([filename])
        
        image, age_label, gender_label = read_and_decode(filename_queue)
        images, age_labels, gender_labels = tf.train.shuffle_batch(
                                                [image, age_label, gender_label], 
                                                batch_size=batch_size,
                                                num_threads=2,
                                                min_after_dequeue=10,
                                                capacity=50000)
        
        return images, tf.reshape(age_labels, [batch_size]), tf.reshape(gender_labels, [batch_size])
 

def split_complex(x):
    # Returns the real and complex parts of x as separate arrays of x.shape
    x_re = tf.real(x)
    x_im = tf.imag(x)
    return x_re, x_im



def conv2d_complex(x, W):
    # Calculates the 2D convolution of x against complex filter W
    # Split into real and imaginary components
    W_re, W_im = split_complex(W)
    
    # Cast into floats required by TensorFlow convolution
    W_re = tf.cast(W_re, tf.float32)
    W_im = tf.cast(W_im, tf.float32)
    
    # Calculate seperate convolution of components
    h_re = tf.nn.conv2d(x, W_re, strides=[1, 1, 1, 1], padding='SAME')
    h_im = tf.nn.conv2d(x, W_im, strides=[1, 1, 1, 1], padding='SAME')
    
    # Create complex output tensor
    return tf.complex( h_re, h_im )
    
    
    
def gabor_filter(input_tensor, s_filters, c_filters, filter_size, La=2.5, Sigma=0.001):
    
    # Begin batch filtering
    start_time = time.time()
    print('INFO: Performing batch filtering...')
    
    # Get filter dimensions
    r, c, s, o = s_filters.get_shape().as_list()
    batch_size, data_row, data_col, data_chan = input_tensor.get_shape().as_list()
        
    for n in range(0, s):
        
        # Select circular filter at index n
        c_filter = c_filters[:,:,n]
        
        # Reshape into TensorFlow standard: [filter_size, filter_size, in_chan, out_chan]
        c_filter = tf.reshape(c_filter, shape=[filter_size, filter_size, 1, 1])
        
        # Calculate Den
        Den = tf.abs( conv2d_complex(input_tensor, c_filter) )
        
        for m in range(0, o):
            
            # Select spatial filter at index (n,m)
            s_filter = s_filters[:,:,n,m]
            
            # Reshape into TensorFlow standard: [filter_size, filter_size, in_chan, out_chan]
            s_filter = tf.reshape(s_filter, shape=[filter_size, filter_size, 1, 1])
            
            # Perform calculations
            top = tf.abs( conv2d_complex(input_tensor, s_filter) )
            bot = tf.exp( tf.divide(-tf.pow(Den, 2), (2 * tf.pow(Sigma, 2)))) + Den
            out = tf.divide(top, bot)
            
            # Perform mean pooling of output tensor. Shape -> [128, 32, 32, 1]
            out_pool = tf.nn.pool(out, window_shape=[2,2], strides=[2,2], pooling_type='AVG', padding='SAME')
                        
            _, s1, s2, _ = out_pool.get_shape().as_list()
            
            
            # Calculate mean of each image in batch. Shape -> [128, 1] i.e. [batch_size, 1]
            mean_tensor = tf.reduce_mean( tf.reduce_mean( tf.abs(out_pool), axis=1 ), axis=1)
            mean_tensor = tf.reshape(mean_tensor, shape=[batch_size, 1, 1, 1])
            
            mu = tf.ones(shape=out_pool.get_shape()) * mean_tensor
            
            shunt = tf.divide( tf.pow(out_pool, La), (tf.pow(out_pool, La) + tf.pow(mu, La)) )

            if n == 0 and m == 0:
                Y = shunt
            else:
                Y = tf.concat([Y, shunt], axis=-1 )
                
    elapsed_time = time.time() - start_time
    print('INFO: Batch filtering took {} s'.format(elapsed_time))
    return Y

def conv_layer(input, params, mode='Test'):
    
    # Calculate if dropout and batch normalization will be applied if network is training
    training = (mode == 'Train')
    
    # Setup a convolution layer using params
    layer = tf.layers.conv2d(inputs=input,
                             filters=params['filters'],
                             kernel_size=params['kernel_size'],
                             padding=params['padding'],
                             activation=params['activation'],
                             name=params['name'])
    
    # Add batch normalization to the layer, else skip
    layer = tf.layers.batch_normalization(layer, training=training) if params['batch_norm'] else layer
    
    # Add dropout to the layer, else skip
    layer = tf.layers.dropout(inputs=layer, rate=params['dropout'], training=training) if params['dropout'] else layer

    # Perform pooling on the output of the layer, else skip
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2, 2], strides=[2, 2]) if params['pool'] else layer
    
    return layer


def dense_layer(input, params, mode='Test'):
               
    # Calculate if dropout and batch normalization will be applied if network is training
    training = (mode == 'Train')

    # Setup a fully connected layer               
    layer = tf.layers.dense(inputs=input,
                              units=params['units'],
                              activation=params['activation'],
                              name=params['name'])

    # Add batch normalization to the layer, else skip
    layer = tf.layers.batch_normalization(layer, training=training) if params['batch_norm'] else layer
    
    # Add dropout to the layer, else skip
    layer = tf.layers.dropout(inputs=layer, rate=params['dropout'], training=training) if params['dropout'] else layer

    return layer


def gabor_layer(input, num_scales, num_orientations, filter_size, f_max):
    
     # Setup filter coefficients using numpy
     spatial = gb.spatial_filter(num_scales=num_scales, num_orientations=num_orientations, filter_size=filter_size,
                                            gamma=1, eta=1, freq_max=f_max)
     circular = gb.circular_filter(num_scales=num_scales, filter_size=filter_size, 
                                              gamma=1, eta=1, freq_max=f_max)
        
     # Convert numpy filters into tensorflow non-trainable variables (i.e. constant filters)
     s_filters = tf.Variable( tf.convert_to_tensor(spatial), trainable=False)
     c_filters = tf.Variable( tf.convert_to_tensor(circular), trainable=False)
        
     # s_filters.shape -> (5, 5, 4, 9) i.e. (size, size, scales, orients)
     # c_filters.shape -> (5, 5, 4) i.e. (size, size, scales)
     # input_layer.shape -> (128, 64, 64, 1) i.e. (batch, height, width, channels)
           
     return gabor_filter(input, s_filters, c_filters, filter_size)



# This function constructs our TensorFlow CNN
def cnn_model_fn(images, mode):
    
    params = []
    
    params.append({'filters': 64, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': False, 'dropout': 0.0, 'name': 'conv1_1'})
    params.append({'filters': 64, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': True, 'dropout': 0.0, 'name': 'conv1_2'})
    params.append({'filters': 128, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': False, 'dropout': 0.1, 'name': 'conv2_1_1'})
    params.append({'filters': 128, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': True, 'dropout': 0.1, 'name': 'conv2_2_1'})
    params.append({'filters': 256, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': False, 'dropout': 0.2, 'name': 'conv3_1_1'})
    params.append({'filters': 256, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': True, 'dropout': 0.2, 'name': 'conv3_2_1'})
    # Gender          
    params.append({'filters': 128, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': False, 'dropout': 0.0, 'name': 'conv2_1_2'})
    params.append({'filters': 128, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': True, 'dropout': 0.0, 'name': 'conv2_2_2'})
    params.append({'filters': 256, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': False, 'dropout': 0.1, 'name': 'conv3_1_2'})
    params.append({'filters': 256, 'kernel_size': [3,3], 'padding': 'same', 'activation': tf.nn.relu, 
          'batch_norm': False, 'pool': True, 'dropout': 0.1, 'name': 'conv3_2_2'})
    
    fc_params = []
    fc_params.append({'units': 1024, 'activation': tf.nn.relu, 'dropout': 0.3, 'batch_norm': False, 'name': 'age_dense1'})
    fc_params.append({'units': 1024, 'activation': tf.nn.relu, 'dropout': 0.4, 'batch_norm': False, 'name': 'age_dense2'})
    fc_params.append({'units': 1024, 'activation': tf.nn.relu, 'dropout': 0.3, 'batch_norm': False, 'name': 'gender_dense1'})
    fc_params.append({'units': 1024, 'activation': tf.nn.relu, 'dropout': 0.4, 'batch_norm': False, 'name': 'gender_dense2'})
    
      
    with tf.name_scope('Input_Layer'):
        # Reshapes the input image tensor into the set size
        input_layer = tf.reshape(images, [-1, 128, 128, 1])
        gender_input = tf.image.resize_images(input_layer, [64, 64])
     
    with tf.name_scope('Gabor_Filter'):
        # First convolutional module
        age_network = gabor_layer(input_layer, 6, 10, 5, 0.6)
        gender_network = gabor_layer(gender_input, 6, 10, 7, 0.6)
       
    with tf.name_scope('Conv_Module_2'):
        age_network = conv_layer(age_network, params[0], mode)
        age_network = conv_layer(age_network, params[1], mode)    
    
    with tf.name_scope('Conv_Module_3_1'):
        age_network = conv_layer(age_network, params[2], mode)
        age_network = conv_layer(age_network, params[3], mode)    
        
    with tf.name_scope('Conv_Module_4_1'):
        age_network = conv_layer(age_network, params[4], mode)
        age_network = conv_layer(age_network, params[5], mode)    
        
    with tf.name_scope('Conv_Module_3_2'):
        gender_network = conv_layer(gender_network, params[6], mode)
        gender_network = conv_layer(gender_network, params[7], mode)    
        
    with tf.name_scope('Conv_Module_4_2'):
        gender_network = conv_layer(gender_network, params[8], mode)
        gender_network = conv_layer(gender_network, params[9], mode)    
    
    with tf.name_scope('Flatten_1'):
        age_network = tf.reshape(age_network, [-1, 8 * 8 * 256])
        gender_network = tf.reshape(gender_network, [-1, 8 * 8 * 256])
    
    with tf.name_scope('Age_FC'):
        age_network = dense_layer(age_network, fc_params[0], mode)
        age_network = dense_layer(age_network, fc_params[1], mode)
        
    with tf.name_scope('Gender_FC'):
        gender_network = dense_layer(gender_network, fc_params[2], mode)
        gender_network = dense_layer(gender_network, fc_params[3], mode)
        
    
    with tf.name_scope('Logits'):
        age_logits = tf.layers.dense(inputs=age_network, units=101, activation=tf.nn.relu, name='age_logits')
        gender_logits = tf.layers.dense(inputs=gender_network, units=2, activation=tf.nn.relu, name='gender_logits')
    
        return age_logits, gender_logits


    
def loss_function(logits, labels):
    
    label_indices = tf.cast(labels, tf.int32)
    
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(label_indices, depth=101),
            logits=logits)
    
    loss_ = tf.reduce_mean(loss)

    return loss_
     
    
    
def train(loss, global_step):
    
    # Train the scnn model
    
    learning_rate = tf.train.exponential_decay(1e-4, global_step=global_step, decay_steps=5000,
                                               decay_rate=0.96, staircase=True)
     
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                 decay=0.98,
                                                 momentum=0.01)
    
    
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    return train_op

    
    
  
    
            
 