# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:16:01 2017

@author: wluckow
"""
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd
import numpy as np
import charembedding
import matplotlib.pyplot as plt

data = pd.read_table('train.tsv')
#EACH SENTENCE HAS BEEN SPLIT INTO MULTIPLE PHRASES USING STANFORD PARSER
#Accuracy to beat is 0.76527

#test.tsv is only for submitting to this expired Kaggle competition: 
#https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
def fix_string(string):
    """
    Make all string characters lowercase
    Remove all symbols from a string
    Remove all extra white space (double spaces, triple, etc) from string
    """
    
    symbols = """+=*&^%~_-0123456789/\!@#$?"'()[]{}|<>`:;,."""
    
    for symbol in symbols:
        string = string.replace(symbol, "") #remove symbols
        
    string_fixed = " ".join(string.split()) #remove white space
    string_fixed = string_fixed.lower() #make lowercase
    
    return string_fixed

# %%
#split the phrases from the labels
def create_arrays(dataframe):
    phrases = dataframe.Phrase.as_matrix()
    for i in range(len(phrases)):
        phrases[i] = fix_string(phrases[i])

    #create an array of labels and one-hot encode them 
    #i.e. [2, 0, 1] --> [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

    labels = dataframe.Sentiment
    labels = pd.get_dummies(labels)
    labels = labels.as_matrix()
    
    return phrases, labels
    
phrases, labels = create_arrays(data)

print(phrases[0])
print(labels[0])

# %%
print(phrases.shape)
print(labels.shape)

# %%
def load_strings_and_labels(phrases, labels):
    """
    takes the list of phrases and encodes them and adds them to a list
    adds the labels to a list
    save the list as csv files
    load the list and return them to numpy arrays
    """
    def save(phrases, labels):
        #encode the data and save it to a large array
        print("Saving...")
        encoded_phrases_array = []
        labels_array = []
        
        for i in range(labels.shape[0]):
            temp_array = charembedding.log_m_embedding(phrases[i], maxlen = 160)
            assert temp_array.shape == (8, 160)
            encoded_phrases_array.append(temp_array)
            
            assert labels[i].shape == (5,)            
            labels_array.append(labels[i])
            
        encoded_phrases_array = np.array(encoded_phrases_array)
        encoded_phrases_array = encoded_phrases_array.reshape(labels.shape[0], 1280)
        labels_array = np.array(labels_array)
        
        phrases_dataframe = pd.DataFrame(encoded_phrases_array)
        labels_dataframe = pd.DataFrame(labels_array)
        
        phrases_dataframe.to_csv('phrases.csv')
        labels_dataframe.to_csv('labels.csv')
        print("Saved.")
        
    #save(phrases, labels) #comment this line out once the csv's have been made
    
    def load(labels):
        print("Loading...")
        X = pd.read_csv("phrases.csv")
        y = pd.read_csv("labels.csv")
        
        X = X.as_matrix()#convert from pandas dataframe to np array
        X = np.delete(X, 0, 1) #remove leftover pandas index
        X = X.reshape(labels.shape[0], 8, 160, 1)
        
        y = y.as_matrix()
        y = np.delete(y, 0, 1)
        
        return X, y
    
    X, y = load(labels)
    print("Finished Loading")
    return X, y
    
    
X, y = load_strings_and_labels(phrases, labels)

# %%
def plot_string(string, save = False):
    """
    input is the string
    function will plot the array
    """
    array = charembedding.log_m_embedding(string)
    #remove axes, extra white space, set dimensions, and save it as test.png
    if save == True:
        save_name = input("Name the file: ")
        save_name = str(save_name)
        try:
            fig = plt.figure(figsize = (8, 0.8), dpi = 100, frameon = False)
            ax = plt.axes([0, 0, 1, 1])
            plt.imshow(array, cmap = 'Greys', interpolation = 'nearest')
            plt.axis('off')    
            plt.savefig('{}.png'.format(save_name))
            plt.close()
        except:
            print("Invalid file name")
    else:
        plt.imshow(array, cmap = 'Greys', interpolation = 'nearest')
        plt.show()

#plot_string(phrases[0], save = False)

# %%

#split the data into train, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 7)

#clear the saved arrays to save space
X = 0
y = 0

print("Training Set:", X_train.shape, y_train.shape)
print("Testing Set:", X_test.shape, y_test.shape)
print("Validation Set:", X_val.shape, y_val.shape)

# %%
#Build and run the convolutional neural network
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
    
learning_rate = 1e-4
epochs = 1000
num_examples = X_train.shape[0]
batch_size = 64
keep_prob_percent = 0.5

tf.reset_default_graph()

model = tf.Graph()
with model.as_default():
    with tf.device('/gpu:0'):
        #Initialize the weights
        W_conv0 = tf.Variable(tf.truncated_normal(shape = [3, 3, 1, 128], stddev = 0.1), name = "W_0")
        W_conv1 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 128], stddev = 0.1), name = "W_1")
        W_conv2 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 128], stddev = 0.1), name = "W_2")
        W_conv3 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 128], stddev = 0.1), name = "W_3")
        W_conv4 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 128], stddev = 0.1), name = "W_4")
        W_conv5 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 128], stddev = 0.1), name = "W_5")
        W_fc0 = tf.Variable(tf.truncated_normal(shape = [1337, 512], stddev = 0.1), name = "W_6")
        W_fc1 = tf.Variable(tf.truncated_normal(shape = [512, 512], stddev = 0.1), name = "W_7")
        W_fc2 = tf.Variable(tf.truncated_normal(shape = [512, 5], stddev = 0.1), name = "W_8")
        
        #Initialize the biases
        b_conv0 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_0")
        b_conv1 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_1")
        b_conv2 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_2")
        b_conv3 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_3")
        b_conv4 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_4")
        b_conv5 = tf.Variable(tf.constant(0.1, shape = [128]), name = "b_5")
        b_fc0 = tf.Variable(tf.constant(0.1, shape = [512]), name = "b_6")
        b_fc1 = tf.Variable(tf.constant(0.1, shape = [512]), name = "b_7")
        b_fc2 = tf.Variable(tf.constant(0.1, shape = [5]), name = "b_8")
        
        #Initialize the input tensors and the keep_prob parameter
        X = tf.placeholder(tf.float32, shape = [None, 8, 160, 1])
        y = tf.placeholder(tf.float32, shape = [None, 5])
        keep_prob = tf.placeholder(tf.float32)        
        
        #Build the graph using the functions, weights, biases, and placeholders
        conv_0 = conv2d(X, W_conv0)
        conv_0 = tf.add(conv_0, b_conv0)
        conv_0 = tf.nn.relu(conv_0)
        
        conv_1 = conv2d(conv_0, W_conv1)
        conv_1 = tf.add(conv_1, b_conv1)
        conv_1 = tf.nn.relu(conv_1)
        
        conv_2 = conv2d(conv_1, W_conv2)
        conv_2 = tf.add(conv_2, b_conv2)
        conv_2 = tf.nn.relu(conv_2)
        
        conv_2 = max_pool_2x2(conv_2)
        
        conv_3 = conv2d(conv_2, W_conv3)
        conv_3 = tf.add(conv_3, b_conv3)
        conv_3 = tf.nn.relu(conv_3)
        
        conv_4 = conv2d(conv_3, W_conv4)
        conv_4 = tf.add(conv_4, b_conv4)
        conv_4 = tf.nn.relu(conv_4)
        
        conv_5 = conv2d(conv_4, W_conv5)
        conv_5 = tf.add(conv_5, b_conv5)
        conv_5 = tf.nn.relu(conv_5)
        
        conv_5 = max_pool_2x2(conv_5)
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        