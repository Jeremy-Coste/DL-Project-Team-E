#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 09:23:59 2018

@author: linye
"""

import numpy as np
import pandas as pd

import os
import time
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import f1_score

class RNNModel():
    
    def __init__(self, learning_rate = 0.0001, keep_prob = 0.5, lambd = 0, n_epoch = 5000, n_batch = 700, display_step = 100,\
                 timesteps = 5, num_hidden = 10, num_classes = 3, num_layers = 2, log_files_path=os.path.join(os.getcwd(), 'logs/')):
        
        self._learning_rate = learning_rate        
        self._keep_prob = keep_prob
        self._lambd = lambd
        self._n_epoch = n_epoch
        self._n_batch = n_batch
        self._display_step = display_step
        self._timesteps = timesteps        
        self._num_hidden = num_hidden
        self._num_classes = num_classes
        self._num_layers = num_layers
        self._log_files_path = log_files_path
        
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        
        self._pred_train = None
        self._pred_test = None
        
    
    def _RNN(self, x, weights, biases):
        
        timesteps = self._timesteps
        num_hidden = self._num_hidden
        keep_prob = self._keep_prob
        num_layers = self._num_layers
        
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, timesteps, 1)
        
        # 1-layer LSTM with num_hidden units
        #rnn_cell = rnn.BasicLSTMCell(num_hidden,activation=tf.nn.sigmoid)
        #rnn_cell = rnn.BasicLSTMCell(num_hidden)   
        
        # 2-layer LSTM, each layer has num_hidden units. And you can wrap more layers together by doing list comprehension.
        #rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden),rnn.BasicLSTMCell(num_hidden)])
        
        rnn_cell = rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.LSTMCell(num_hidden), output_keep_prob=keep_prob) for _ in range(num_layers)])
        
        # Get rnn cell output
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
        
        # (1) Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    def Train(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        
        timesteps = self._timesteps
        num_hidden = self._num_hidden
        num_classes = self._num_classes
        n_batch = self._n_batch
        lambd = self._lambd
        learning_rate = self._learning_rate     
        n_epoch = self._n_epoch
        display_step = self._display_step
        log_files_path = self._log_files_path
        num_input = x_train.shape[-1]
        
        # tf Graph input
        # the input variables are first define as placeholder
        # a placeholder is a variable/data which will be assigned later
        X = tf.placeholder("float", [None, timesteps, num_input]) #dim: batch_size, number of time steps, number of inputs
        Y = tf.placeholder("float", [None, num_classes])#dim:batch_size, number of classes (10 here)
        
        #initialize the weigths with a normal random law initialization
        weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
        biases = {'out': tf.Variable(tf.random_normal([num_classes]))}
        
        logits = self._RNN(X, weights, biases)
        prediction = tf.nn.softmax(logits)        
        
        #define the loss 
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
        
        # add L2 regularization too all weights
        l2 = lambd * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        loss_op = loss_op + l2
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        #tf.equal: Returns the truth value of (x == y) element-wise.
        #tf.cast: Casts a tensor to a new type. --- here it casts from boolean to float
        #tf.argmax:Returns the index with the largest value across axes of a tensor. --- here along the axis of the vector
        
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()
        
        # Start training
        with tf.Session() as sess:   
            
            start_time = time.time()  
            #please, make sure you changed for your own path 
            #log_files_path = '/Users/macbook/Desktop/DLFall2018/codes/DL project/tensorflow-master/logs'
            #log_files_path = '/Users/meihuaren/personal/DL_logs/intc_up&down_nolesssprd_try/'
            
            #save and restore variables to and from checkpoints.
            saver = tf.train.Saver()
            
            # Run the initializer
            sess.run(init)   
            
            #will work with this later
            #saver.restore(sess, log_files_path+'multi_layer/model-checkpoint-66000')
            
            loss_trace = []
            
            num_examples = x_train.shape[0]
            
            # Training cycle
            for epoch in range(n_epoch):
                
                avg_cost = 0.
                total_batch = int(num_examples/n_batch)
                
                # Shuffle the data
                np.random.seed(4720)
                perm = np.arange(num_examples)
                np.random.shuffle(perm)
                x_train = x_train[perm]
                y_train = y_train[perm]
                
                # Loop over all batches
                
                for i in range(total_batch):           
                    
                    minibatch_x, minibatch_y = x_train[i*n_batch:(i+1)*n_batch], y_train[i*n_batch:(i+1)*n_batch]
                    #minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                    
                    # Fit training using batch data
                    #the training is done using the training dataset
                    sess.run(train_op, feed_dict={X: minibatch_x, Y: minibatch_y})
                    # Compute average loss
                    avg_cost += sess.run(loss_op, feed_dict={X: minibatch_x, Y: minibatch_y})/total_batch
                
                # Display logs per epoch step
                if epoch % display_step == 0:                   
                    
                    #the accuracy is evaluated using the validation dataset
                    train_acc = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})
                    val_cost = sess.run(loss_op, feed_dict={X: x_valid, Y: y_valid})
                    acc = sess.run(accuracy, feed_dict={X: x_valid, Y: y_valid})
                    loss_trace.append(1-acc)    
                    print("Epoch:", '%04d' % (epoch+1), "train_loss :", "{:0.4f}".format(avg_cost), "train_acc :", "{:0.4f}".format(train_acc), \
                          "val_loss :", "{:0.4f}".format(val_cost), "val_acc :", "{:0.4f}".format(acc))    
            
            #save to use later
            saver.save(sess, log_files_path)
            
            print("Optimization Finished!")
            #accuracy evaluated with the whole test dataset
            acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
            print("Test Accuracy:", acc)
            
            elapsed_time = time.time() - start_time
            print('Execution time (seconds) was %0.3f' % elapsed_time)
        
            self._pred_train = sess.run(prediction, feed_dict={X: x_train, Y: y_train}) 
            self._pred_test = sess.run(prediction, feed_dict={X: x_test, Y: y_test})
            
            self._x_train = x_train
            self._x_test = x_test
            self._y_train = y_train
            self._y_test = y_test
        
        tf.reset_default_graph()
    
    def Evaluate(self, result_path = os.path.join(os.getcwd(), 'data')):
        
        pred_df_train = pd.DataFrame(self._pred_train, columns=['-1', '1'])
        pred_df_train['predict'] = pred_df_train.idxmax(axis=1)
        pred_df_train['true'] = pd.DataFrame(self._y_train, columns=['-1','1']).idxmax(axis=1)
            
        #filename = os.path.join(result_path, 'pred_train_intc_up&down_nolesssprd.csv')
        #pred_df_train.to_csv(filename, index=False)
            
        pred_df_test = pd.DataFrame(self._pred_test, columns=['-1', '1'])
        pred_df_test['predict'] = pred_df_test.idxmax(axis=1)
        pred_df_test['true'] = pd.DataFrame(self._y_test, columns=['-1', '1']).idxmax(axis=1)  
            
        filename = os.path.join(result_path, 'pred_test_intc_up&down_nolesssprd.csv')
        pred_df_test.to_csv(filename, index=False)
        
        train_f1 = f1_score(pred_df_train.true, pred_df_train.predict, average=None)
        test_f1 = f1_score(pred_df_test.true, pred_df_test.predict, average=None)
        print (train_f1)
        print (test_f1)
        
        return pred_df_train, pred_df_test
        
        
    

        
        