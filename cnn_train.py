#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
cnn_train.py

Author: Jacob Bennett
Contact: jab999@uowmail.edu.au
Date Created: 10/04/2018
Last Updated: 19/04/2018

This module was created as an alternative to using the high level Estimator API in TensorFlow. This
allows much more control over program flow, evaluation runs, automatic monitoring of training, and
easier export of variables such as accuracy, loss for presenting results.

Example
----------
To run this script from the command line, activate the tensorflow environment then use the standard
python syntax in running scripts:

    $ source activate tensorflow_env
    $ (tensorflow_env) python cnn_train.py
    
Attributes
----------
   


'''

# Important library for datapath management
import os

# Import our TensorFlow model
import cnn_model

# Import Time to name output
import time

# Tensorflow libraries to allow training and exporting of our CNN model
import tensorflow as tf

# Helps in debugging as it displays the saved variables in a checkpoint
from tensorflow.python.tools import inspect_checkpoint as chpk




TRAIN_FILE = 'Spring/IMDB/No_Proc/train_128_equal_gray.tfrecords'
TEST_FILE = 'Spring/IMDB/No_Proc/test_128_equal_gray.tfrecords'
VAL_FILE = 'Spring/LFW/lfw_128_gray.tfrecords'
AGE_VAL_FILE = 'Spring/CHALEARN/test_chalearn_128.tfrecords'

# Image batch info
BATCH_SIZE = 128
IMG_SIZE = 128   # Assume height = width
IMG_CHAN = 1

# Number of training images
TRAIN_DATA_SIZE = 137718    # IMDB-Wiki
TEST_DATA_SIZE = 15304
VAL_DATA_SIZE = 13233       # LFW
UOW_DATA_SIZE = 40000
#TRAIN_DATA_SIZE = 60000
#VAL_DATA_SIZE = 20000

# Path to initialization vars
INIT_PATH = 'init_60_128_256_1024'



def check_data(data_dir, train_file=None, eval_file=None, validation_file=None):
    '''
    check_data checks the directory passed in 'data_dir' and checks if any of the passed data files
    exist
    
    Parameters
    ------------
    data_dir : string
            location of the parent directory holding the data
    train_file : string
            name of training data path within parent directory
    eval_file : string
            name of the evaluation data path within parent directory
    validation_file : string 
            name of the validation data path within parent directory
            
    Returns
    -----------
    Bool : true if data does not exist (there are errors), false if data exists
    
    '''
    
    data_exist = []
    try:
        if train_file is not None:
            data_exist.append(os.path.isfile(os.path.join(data_dir, train_file)))
        if eval_file is not None:
            data_exist.append(os.path.isfile(os.path.join(data_dir, eval_file)))
        if validation_file is not None:
            data_exist.append(os.path.isfile(os.path.join(data_dir, validation_file)))
        if not any(data_exist):
            raise Exception('INFO: Data error')
    except:        
        print('INFO: Data error')
        print('INFO: Cancelling TensorFlow execution')
        return True
    
    print('INFO: Data found succesfully')
    return False
    

    

def get_output_dir(output_root_dir):
    '''
    Check the current output directory for existing TensorFlow models to train/evaluate
    
    Parameters
    ------------
    output_root_dir : string
        parent directory containing TensorFlow output folders
         
    Returns
    -----------
    string
        output folder to load existing TensorFlow model from
        
    '''
    
    outputs = []
    num = 1
    print('INFO: Saved models in current directory:')
    for root, dirs, files in os.walk(output_root_dir):
        for dirname in sorted(dirs):
            if 'output' in dirname:
                outputs.append(dirname)
                print('[{}] {}'.format(num, dirname))
                num += 1
                
    choice = int(raw_input('INFO: Enter saved model to use [1-{}]: '.format(num-1)))-1
    return outputs[choice]



def log_eta(elapsed_time, current_step, steps_per_epoch, epoch, num_epochs):
    '''
    Returns the current eta to complete the current epoch
    
    Parameters
    ------------
    
         
    Returns
    -----------
    
        
    '''
    
    # Calculate remaining steps for the current epoch
    steps_remaining = steps_per_epoch - current_step
    
    # Elapsed time is for 100 steps, so divide steps remaining by 100 then multiply by elapsed time
    time_remaining = steps_remaining / 100.0 * elapsed_time 
    
    # Estimate time taken for a single complete epoch (Plus 10% for evaluation ops)
    epoch_time = steps_per_epoch / 100.0 * elapsed_time * 1.05
    
    # Estimate remaining time over all epochs
    eta = (num_epochs - epoch) * epoch_time + time_remaining
    
    tot_minutes = eta / 60.0
    hours_left = int(tot_minutes / 60)
    minutes_left = tot_minutes - hours_left * 60
    
    
    print('INFO: ETA: {} h, {:.2f} m'.format(hours_left, minutes_left))
    
    return
   
    

def log_accuracy(filename, accuracy, step, epoch, loss):
    '''
    Logs the accuracy to a .txt file
    '''
    
    fp = open(filename, 'a')   
    fp.write('epoch: {}, step: {}, loss: {}, accuracy: {}\n'.format(epoch, step, loss, accuracy))  
    fp.close()


def train(data_dir, output_dir, num_epochs):
    
    '''
    Trains the CNN model described in cnn_model.py for a specified amount of epochs. By specifying
    the output directory, an existing model can be loaded by the tf.train.Saver() API.
    
    Parameters
    ------------
    data_dir : string
            location of the parent directory holding the data files
    output_dir : string
            name of output path to store logs and model checkpoints after training
    num_epochs : int
            amount of epochs to run model for (how many times to cycle the entire training dataset)

    Returns
    -----------
    None
    
    '''
    
    # Train a SCNN model
    cnn_graph = tf.Graph()
    
    # Set start time
    start_time = time.time()
    last_time = start_time
    
    # Set up epochs and step size
    steps_per_epoch = int(TRAIN_DATA_SIZE / BATCH_SIZE)
     
    # Generate absolute train file path
    train_file = os.path.join(data_dir, TRAIN_FILE)
    test_file = os.path.join(data_dir, TEST_FILE)
    val_file = os.path.join(data_dir, VAL_FILE)
    age_val_file = os.path.join(data_dir, AGE_VAL_FILE)


    with cnn_graph.as_default():
        
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels
        images, age_labels, gender_labels = cnn_model.input_fn(train_file, batch_size=BATCH_SIZE)
        
        # Build CNN graph
        age_logits, gender_logits = cnn_model.cnn_model_fn(images, mode='Train')

        # Training batch accuracy
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        age_loss = tf.losses.absolute_difference(age_labels, age)
        
        gender_ = tf.losses.softmax_cross_entropy(
                            onehot_labels=tf.one_hot(gender_labels, depth=2),
                            logits=gender_logits)
    
        gender_loss = tf.reduce_mean(gender_)
        
        
        total_loss = tf.add_n([age_loss, gender_loss])
        
        train_op = cnn_model.train(total_loss, global_step)
        
        # Create session        
        with tf.Session(graph=cnn_graph) as sess:
            
            # TODO: configure summaries to write
            
            # Initialize all variables
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            
            # Setup saver to control model checkpoints
            saver = tf.train.Saver()
            
            # Here we load the most recent checkpoint if it is available
            ckpt = tf.train.get_checkpoint_state(output_dir)
            
            # If checkpoint and checkpoint path exist, then restore model
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)   
                
                # Get global step from checkpoint model
                global_step = tf.train.get_or_create_global_step()
                
                # Used to update training step in loop
                # Calculate starting epoch
                start_epoch = int(global_step.eval()/steps_per_epoch) + 1
                # Calculate starting step
                start_step = global_step.eval() - steps_per_epoch * (start_epoch-1)
            else:
                # Use default starting values
                start_epoch = 1
                start_step = 1
                
                # Load variable init checkpoint from TF_Init folder
                #print('INFO: New Model - Loading initialization checkpoint')
                #ckpt = tf.train.get_checkpoint_state(INIT_PATH)
                #chpk.print_tensors_in_checkpoint_file(ckpt.model_checkpoint_path, all_tensors=True, tensor_name='')
                #saver.restore(sess, ckpt.model_checkpoint_path)   
                
            
            # Start input pipeline
            coord = tf.train.Coordinator()
            
            # Set up queue threads
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            begin_training = time.time()
            
            # Variable to monitor top performing model
            top_accuracy = 0.0
            
            for epoch in range(start_epoch, num_epochs+1):
                
                for local_step in range(start_step, steps_per_epoch+1):
                                                            

                    # Feed image batch into model and get training loss and predictions
                    _, age_out, gender_out = sess.run([train_op, age_loss, gender_loss])
                    
                    # Every 100 steps log loss info to terminal
                    if not global_step.eval() % 100:
                        elapsed_time = time.time() - last_time
                        print('INFO: epoch = {}/{}, step = {}, Gender Loss = {}, Age Loss = {} ({:.4f} sec)'.format(epoch, num_epochs, global_step.eval(), gender_out, age_out, elapsed_time))
                        
                        # TODO: update this to work with global step
                        log_eta(elapsed_time, local_step, steps_per_epoch, epoch, num_epochs)
                        last_time = time.time()
                        
                       
                    # Every 1000 steps, calculate training accuracy and save checkpoint
                    if not global_step.eval() % 1000:                    
                        
                        saver.save(sess, os.path.join(output_dir, output_dir), global_step=global_step)
                        
                        # Keep track of how many checkpoints have been saved                        
                        num_checkpoints += 1
                        
                        age_eval_acc, gender_eval_acc = run_eval(test_file, output_dir, data_size=TEST_DATA_SIZE, batch_size=128)
                        
                        try:
                            log_accuracy(os.path.join(output_dir, 'eval_accuracy_age.txt'), age_eval_acc, global_step.eval(), epoch, 0)
                            log_accuracy(os.path.join(output_dir, 'eval_accuracy_gender.txt'), gender_eval_acc, global_step.eval(), epoch, 0)
                        except:
                            print('INFO: Error writing accuracy logs to file')
                            return
                     
                # Save epoch      
                saver.save(sess, os.path.join(output_dir, output_dir), global_step=global_step)
                
                if not epoch % 1:        
                    # Run evalutation for train dataset every batch
                    _, gender_eval_acc = run_eval(val_file, output_dir, data_size=VAL_DATA_SIZE, batch_size=128)
                    age_eval_acc, _ = run_eval(age_val_file, output_dir, data_size=1978, batch_size=128)
                    
                    #uow_accuracy = run_eval(uow_file, output_dir, data_size=UOW_DATA_SIZE, batch_size=1)
                
                    try:
                        log_accuracy(os.path.join(output_dir, 'lfw_accuracy.txt'), gender_eval_acc, local_step, epoch, gender_loss)
                        log_accuracy(os.path.join(output_dir, 'chalearn_accuracy.txt'), age_eval_acc, local_step, epoch, gender_loss)
                        #log_accuracy(os.path.join(output_dir, 'uow_accuracy.txt'), uow_accuracy, local_step, epoch, loss_out)
                    except:
                        print('INFO: Error writing accuracy logs to file')
                        return
                    
                    if gender_eval_acc >= top_accuracy:
                        # Save top accuracy models in separate folder
                        saver.save(sess, os.path.join(output_dir, 'top_perform/' + output_dir), global_step=global_step)
                        top_accuracy = gender_eval_acc
                   
                # Reset starting step as new model is not loaded
                start_step = 1
                
                print('INFO: Epoch {} completed'.format(epoch))
                
        
            coord.request_stop()
            coord.join(threads)
            sess.close()
            
            end_training = time.time() - begin_training
            print('INFO: Total time for {} epochs: {} min'.format(num_epochs, end_training/60))
            
            return
        

def run_eval(eval_path, output_dir, data_size, batch_size):
    '''
    Evaluates a trained model checkpoint specified in the output_dir. By default, the newest 
    checkpoint is used but this may be changed at a later date.
    
    Parameters
    ------------
    eval_path : string
        path to the evaluation dataset in memory    
    output_dir : string
            location of the output directory containing the model to evaluate
    data_size : int
            size of data to be evaluated.
    batch_size : int
            batch size of evaluation data

    Returns
    -----------
    int
        accuracy over entire dataset of `data_size' amount of images
    
    '''
    
    eval_graph = tf.Graph()
    
     
    with eval_graph.as_default():
        # Calculate max step
        max_eval_steps = int(data_size/batch_size)
        
        # TODO: return filepath from input function to view incorrect predictions
        
        # Get images and labels
        images, age_labels, gender_labels = cnn_model.input_fn(eval_path, batch_size=batch_size)
        

        # Build CNN graph
        age_logits, gender_logits = cnn_model.cnn_model_fn(images, mode='Test')
        
        # Training batch accuracy
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        age_error = tf.losses.absolute_difference(age_labels, age)
        
        gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))               
            
                
        with tf.Session(graph=eval_graph) as sess:
                
            # Initialize cnn variables
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
                
            # Setup saver
            saver = tf.train.Saver()
            
            # Restore checkpoint for evaluation
            ckpt = tf.train.get_checkpoint_state(output_dir)
                
            # If checkpoint and checkpoint path exist, then restore model
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                
            # Start input pipeline
            coord = tf.train.Coordinator()
                    
            # Set up queue threads
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            # Setup accuracy variable
            age_sum = 0
            gender_sum = 0
            
            # Run over evaluation dataset
            for eval_step in range(1, max_eval_steps+1):
                
                # Print eval step every 100 steps
                if max_eval_steps > 10000 and not eval_step % 1000:
                    print('INFO: Running evaluation step {}/{}'.format(eval_step, max_eval_steps))
                    
                if max_eval_steps < 1000:
                    print('INFO: Running evaluation step {}/{}'.format(eval_step, max_eval_steps))
                
                # Calculate accuracy
                age_out, gender_out = sess.run([age_error, gender_acc])
                
                age_sum += age_out
                gender_sum += gender_out
                
            age_eval_error = age_sum / max_eval_steps
            gender_eval_error = gender_sum / max_eval_steps
            
            print('INFO: Age evaluation absolute error: {}'.format(age_eval_error))
            print('INFO: Gender evaluation accuracy: {}'.format(gender_eval_error))
            
            coord.request_stop()
            coord.join(threads)
            sess.close()
            
            return age_eval_error, gender_eval_error
  
    
    

                                                

def main():
    
    '''
    Main function, sets up output directories and calls training function
    
    Parameters
    ------------
    None

    Returns
    -----------
    None
    
    '''
    
    # Set up datapath and output directories
    data_dir = '/media/jacob/LaCie/2018/Data/Face_Data/TFRecords'   
    
    # Check for valid data
    if check_data(data_dir, train_file=TRAIN_FILE, eval_file=TEST_FILE) : return
    
    # Setup new output directory if training new model, else get user to select existing model
    if( raw_input('INFO: Train new model (y/n): ') != 'y'):
        output_dir = get_output_dir(os.getcwd())
    else:
        tags = raw_input('Enter any useful labels for model (no spaces): ')
        output_dir = time.strftime('output_%d%b_%H%M%S_{}'.format(tags), time.localtime())
     
    # Print directories to be used
    print('INFO: Dataset path set as: {}'.format(data_dir))
    print('INFO: Output path set as: {}'.format(output_dir))
    
    # Print ready message, data and directories have been set up
    if( raw_input('INFO: Continue with TensorFlow execution (y/n): ') != 'y'):
        print('INFO: TensorFlow execution cancelled by user')
        print('INFO: Cancelling TensorFlow execution')
        return
    
    # Train CNN
    train(data_dir, output_dir, num_epochs=20)




if __name__ == '__main__':
    # Run main()
    main()
