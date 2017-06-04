# -*- coding: utf-8 -*-
"""
Created on Wed May 31 00:17:15 2017

@author: wluckow
"""

# %%
# Import packages
#keep adding packages as needed and re-running this cell
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import glob
from collections import Counter
import charembedding
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
from time import perf_counter as timer

# %%
#import data
neg_review = glob.glob("W:/Projects/Sentiment Analysis/review_polarity/txt_sentoken/neg/*.txt")
pos_review = glob.glob("W:/Projects/Sentiment Analysis/review_polarity/txt_sentoken/pos/*.txt")

# %%
def make_review_list(neg_review, pos_review):
    """
    input is a list of all the different filenames for the reviews
    output is lists containing all reviews as strings
    """
    
    neg_list = []
    pos_list = []
    symbols = """+=*&^%~_-0123456789/\!@#$?"'()[]{}|<>`:;,."""
    
    for review in neg_review:
        with open(review, 'r') as myfile:
            text_review = myfile.read()
            for char in symbols: #remove special characters
                text_review = text_review.replace(char, "")
            text_review = " ".join(text_review.split()) #remove double spaces, line breaks, etc
            neg_list.append(text_review)
    
    for review in pos_review:
        with open(review, 'r') as myfile:
            text_review = myfile.read()
            for char in symbols: #remove special characters
                text_review = text_review.replace(char, "")
            text_review = " ".join(text_review.split()) #remove double spaces, line breaks, etc
            pos_list.append(text_review)
    return neg_list, pos_list
    
neg_list, pos_list = make_review_list(neg_review, pos_review)

# %%
#create the n most common words in the negative reviews and the positive reviews
def create_common_uncommon_words(neg_list, pos_list, number_of_common_words, number_of_uncommon_words):
    """
    inputs: a list of all the negative reviews, positive reviews, and the
    number of most common words to include. 
    
    returns a list of the n most common words in the negative reviews and 
    the positive reviews. Also returns the m least common words
    """
    
    neg_string = ""
    for i in range(1000):
        neg_string = neg_string + " " + neg_list[i]
    total_neg = neg_string.split()
    
    pos_string = ""
    for i in range(1000):
        pos_string = pos_string + " " + pos_list[i]
    total_pos = pos_string.split()
    
    neg_count = Counter(total_neg)
    pos_count = Counter(total_pos)
    
    neg_words_common = []
    neg_words_uncommon = []
    pos_words_common = []
    pos_words_uncommon = []
    
    neg_words_list = neg_count.most_common()
    neg_words_list_flip = neg_words_list[::-1]
    
    pos_words_list = pos_count.most_common()
    pos_words_list_flip = pos_words_list[::-1]
    
    for i in number_of_common_words:
        neg_words_common.append(neg_words_list[i][0])
        pos_words_common.append(pos_words_list[i][0])
        
    for i in number_of_uncommon_words:
        neg_words_uncommon.append(neg_words_list_flip[i][0])
        pos_words_uncommon.append(pos_words_list_flip[i][0])

    return neg_words_common, pos_words_common, neg_words_uncommon, pos_words_uncommon
    
neg_words_common, pos_words_common, neg_words_uncommon, pos_words_uncommon = create_common_uncommon_words(neg_list, pos_list, range(50), range(500))

neg_omit = neg_words_common + neg_words_uncommon
pos_omit = pos_words_common + pos_words_uncommon

# %%
#remove the most common and least common words from all the reviews

for negative_review in range(len(neg_list)):
    
    negative_review_words = neg_list[negative_review].split()
    for words_in_review in range(len(negative_review_words)):
        if negative_review_words[words_in_review] in neg_omit:  
            negative_review_words[words_in_review] = ""    
    neg_list[negative_review] = " ".join(negative_review_words)
    neg_list[negative_review] = " ".join(neg_list[negative_review].split())
    
for positive_review in range(len(pos_list)):
    positive_review_words = pos_list[positive_review].split()
    for words_in_review in range(len(positive_review_words)):
        if positive_review_words[words_in_review] in pos_omit:
            positive_review_words[words_in_review] = ""
    pos_list[positive_review] = " ".join(positive_review_words)
    pos_list[positive_review] = " ".join(pos_list[positive_review].split())
    
# %%
#print(neg_list[0])
print("")
#print(pos_list[0])

# %%
#make the labels
labels = []
for i in range(1000):
    labels.append([0, 1])
for i in range(1000):
    labels.append([1, 0])
reviews = neg_list + pos_list
labels = np.array(labels)
#labels = labels.reshape(labels.shape[0], 1)
    
# %%

def create_encoded_arrays(reviews, labels):
    """
    takes the list of reviews and encodes them and adds them to a list
    adds the labels to a list
    save the list as csv files
    load the list and return them to numpy arrays
    """
        #encode the data and save it to a large array
    encoded_reviews_array = []
    labels_array = []
    
    for i in range(labels.shape[0]):
        temp_review = reviews[i][-500:]
        temp_array = charembedding.log_m_embedding(temp_review, maxlen = 500)
        assert temp_array.shape == (7, 500)
        encoded_reviews_array.append(temp_array)
        
        assert labels[i].shape == (2,)            
        labels_array.append(labels[i])
        
    encoded_reviews_array = np.array(encoded_reviews_array)
    encoded_reviews_array = encoded_reviews_array.reshape(labels.shape[0], 7, 500, 1)
    labels_array = np.array(labels_array)
        
    return encoded_reviews_array, labels_array
    
X, y = create_encoded_arrays(reviews, labels)

# %%
print(X.shape)
print(y.shape)

#split into train, val, test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 7)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 7)

# %%

print("Training Set:", X_train.shape, y_train.shape)
print("Testing Set:", X_test.shape, y_test.shape)
print("Validation Set:", X_val.shape, y_val.shape)

# %%
#Build and run the convolutional neural network
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    
learning_rate = 0.00001
epochs = 1000
num_examples = X_train.shape[0]
batch_size = 512
keep_prob_percent = 0.5

tf.reset_default_graph()

model = tf.Graph()
with model.as_default():
    with tf.device('/gpu:0'):
        #Initialize the weights
        W_conv0 = tf.Variable(tf.truncated_normal(shape = [8, 3, 1, 32], stddev = 0.1), name = "W_0")
        W_conv1 = tf.Variable(tf.truncated_normal(shape = [2, 2, 32, 64], stddev = 0.1), name = "W_1")
        W_conv2 = tf.Variable(tf.truncated_normal(shape = [1, 1, 64, 128], stddev = 0.1), name = "W_2")
        #W_conv3 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 128], stddev = 0.1), name = "W_3")
        #W_conv4 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 128], stddev = 0.1), name = "W_4")
        #W_conv5 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 128], stddev = 0.1), name = "W_5")
        W_fc0 = tf.Variable(tf.truncated_normal(shape = [8064, 512], stddev = 0.1), name = "W_6")
        W_fc1 = tf.Variable(tf.truncated_normal(shape = [512, 512], stddev = 0.1), name = "W_7")
        W_fc2 = tf.Variable(tf.truncated_normal(shape = [512, 2], stddev = 0.1), name = "W_8")
        
        #Initialize the biases
        b_conv0 = tf.Variable(tf.constant(0.1, shape = [32]), name = "b_0")
        b_conv1 = tf.Variable(tf.constant(0.1, shape = [64]), name = "b_1")
        b_conv2 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_2")
        #b_conv3 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_3")
        #b_conv4 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_4")
        #b_conv5 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_5")
        b_fc0 = tf.Variable(tf.constant(0.1, shape = [512]), name = "b_6")
        b_fc1 = tf.Variable(tf.constant(0.1, shape = [512]), name = "b_7")
        b_fc2 = tf.Variable(tf.constant(0.1, shape = [2]), name = "b_8")
        
        #Initialize the input tensors and the keep_prob parameter
        X = tf.placeholder(tf.float32, shape = [None, 7, 500, 1])
        y = tf.placeholder(tf.float32, shape = [None, 2])
        keep_prob = tf.placeholder(tf.float32)        
        
        #Build the graph using the functions, weights, biases, and placeholders
        conv_0 = conv2d(X, W_conv0)
        conv_0 = tf.add(conv_0, b_conv0)
        conv_0 = tf.nn.relu(conv_0)
        
        conv_0 = max_pool_2x2(conv_0)
        
        conv_1 = conv2d(conv_0, W_conv1)
        conv_1 = tf.add(conv_1, b_conv1)
        conv_1 = tf.nn.relu(conv_1)
        
        conv_1 = max_pool_2x2(conv_1)
        
        conv_2 = conv2d(conv_1, W_conv2)
        conv_2 = tf.add(conv_2, b_conv2)
        conv_2 = tf.nn.relu(conv_2)
        
        #conv_2 = max_pool_2x2(conv_2)
        
        #conv_3 = conv2d(conv_2, W_conv3)
        #conv_3 = tf.add(conv_3, b_conv3)
        #conv_3 = tf.nn.relu(conv_3)
        
        #conv_4 = conv2d(conv_3, W_conv4)
        #conv_4 = tf.add(conv_4, b_conv4)
        #conv_4 = tf.nn.relu(conv_4)
        
        #conv_5 = conv2d(conv_4, W_conv5)
        #conv_5 = tf.add(conv_5, b_conv5)
        #conv_5 = tf.nn.relu(conv_5)
        
        conv_5 = max_pool_2x2(conv_2) #CHANGE THIS BACK TO CONV_5
        conv_5_shape = conv_5.get_shape().as_list()        
        
        conv_5 = tf.reshape(conv_5, [-1, conv_5_shape[1] * conv_5_shape[2] * conv_5_shape[3]])
        
        fc_0 = tf.add(tf.matmul(conv_5, W_fc0), b_fc0)
        fc_0 = tf.nn.relu(fc_0)
        fc_0 = tf.nn.dropout(fc_0, keep_prob)
        
        fc_1 = tf.add(tf.matmul(fc_0, W_fc1), b_fc1)
        fc_1 = tf.nn.relu(fc_1)
        fc_1 = tf.nn.dropout(fc_1, keep_prob)
        
        fc_2 = tf.add(tf.matmul(fc_1, W_fc2), b_fc2)
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc_2, labels = y))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
        
        correct_prediction = tf.equal(tf.argmax(fc_2, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        init = tf.global_variables_initializer()
        
tf.reset_default_graph()

def save_model():
    save_file = './train_model.ckpt'
    saver = tf.train.Saver({"W_0": W_conv0,
                            "W_1": W_conv1,
                            "W_2": W_conv2,
                            #"W_3": W_conv3,
                            #"W_4": W_conv4,
                            #"W_5": W_conv5,
                            "W_6": W_fc0,
                            "W_7": W_fc1,
                            "W_8": W_fc2,
                            "b_0": b_conv0,
                            "b_1": b_conv1,
                            "b_2": b_conv2,
                            #"b_3": b_conv3,
                            #"b_4": b_conv4,
                            #"b_5": b_conv5,
                            "b_6": b_fc0,
                            "b_7": b_fc1,
                            "b_8": b_fc2})
    return saver, save_file
    
def train(tfgraph, tfepochs, tfbatch, tfdropout, xtrain, ytrain, xval, yval, xtest, ytest, saver, save_file):
    starter = timer()
    
    with tf.Session(graph = tfgraph) as sess:
        sess.run(init)
        
        for epoch in range(tfepochs):
            
            shuff_X_train, shuff_y_train = shuffle(xtrain, ytrain)
            
            #train
            for offset in range(0, num_examples, tfbatch):
                end = offset + tfbatch
                X_batch, y_batch = shuff_X_train[offset:end], shuff_y_train[offset:end]
                sess.run(optimizer, feed_dict = {X: X_batch, y: y_batch, keep_prob: tfdropout})
            

            shuff_X_val, shuff_y_val = shuffle(xval, yval)
            validation_accuracy = sess.run(accuracy, feed_dict = {X: shuff_X_val, y: shuff_y_val, keep_prob: 1.})
            
            #print the accuracy
            sys.stdout.write("\r" + "Epoch: " + str(epoch) + " ||| Validation Accuracy: " + str(validation_accuracy))
            sys.stdout.flush()
                
        saver.save(sess, save_file)
        print("")
        print("Trained Model Saved.")
        
        #check the final testing accuracy
        shuff_X_test, shuff_y_test = shuffle(xtest, ytest)
        testing_accuracy = sess.run(accuracy, feed_dict = {X: shuff_X_test, y: shuff_y_test, keep_prob: 1.})
        print("")
        print("Testing Accuracy:", testing_accuracy)
        
        ender = timer()
        print("")
        print("Time:", ender - starter)
        
train(model, epochs, batch_size, keep_prob_percent, X_train, y_train, X_val, y_val, X_test, y_test, save_model()[0], save_model()[1])

tf.reset_default_graph

# %%
def test_accuracy():
    with tf.Session(graph = model) as sess:
        save_model()[0].restore(sess, save_model()[1])
        feed_dict = {X: X_test, y: X_test, keep_prob: 1.}
        
        file_writer = tf.summary.FileWriter('./logs/model_graph', sess.graph)
        
        print("Test Accuracy:", accuracy.eval(feed_dict = feed_dict))
    
    tf.reset_default_graph()
    
#test_accuracy()