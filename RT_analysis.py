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
import sys
from time import perf_counter as timer

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
def create_string_arrays(dataframe):
    phrases = dataframe.Phrase.as_matrix()
    for i in range(len(phrases)):
        phrases[i] = fix_string(phrases[i])

    #create an array of labels and one-hot encode them
    #i.e. [2, 0, 1] --> [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

    labels = dataframe.Sentiment
    labels = pd.get_dummies(labels)
    labels = labels.as_matrix()

    return phrases, labels

phrases, labels = create_string_arrays(data)

print(phrases[0])
print(labels[0])

# %%
print(phrases.shape)
print(labels.shape)

# %%
def create_encoded_arrays(phrases, labels):
    """
    takes the list of phrases and encodes them and adds them to a list
    adds the labels to a list
    save the list as csv files
    load the list and return them to numpy arrays
    """
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
    encoded_phrases_array = encoded_phrases_array.reshape(labels.shape[0], labels.shape[1], labels.shape[2], 1)
    labels_array = np.array(labels_array)

    return encoded_phrases_array, labels_array

X, y = create_encoded_arrays(phrases, labels)

# %%
from img_helpers import save_embedding_image

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
            save_embedding_image(array, save_name, (8, 0.8))
        except:
            print("An Error has occured. Is your file name valid?")
    else:
        plt.imshow(array, cmap = 'Greys', interpolation = 'nearest')
        plt.show()

#plot_string(phrases[0], save = False)

# %%

#split the data into train, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 7)

#clear the saved arrays to save space
X = 0
y = 0

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

learning_rate = 1e-4
epochs = 100
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
        W_fc0 = tf.Variable(tf.truncated_normal(shape = [10240, 512], stddev = 0.1), name = "W_6")
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

        init = tf.global_variables_initializer()

tf.reset_default_graph()

def save_model():
    save_file = 'train_model.ckpt'
    saver = tf.train.Saver({"W_0": W_conv0,
                            "W_1": W_conv1,
                            "W_2": W_conv2,
                            "W_3": W_conv3,
                            "W_4": W_conv4,
                            "W_5": W_conv5,
                            "W_6": W_fc0,
                            "W_7": W_fc1,
                            "W_8": W_fc2,
                            "b_0": b_conv0,
                            "b_1": b_conv1,
                            "b_2": b_conv2,
                            "b_3": b_conv3,
                            "b_4": b_conv4,
                            "b_5": b_conv5,
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
            validation_accuracy = sess.run(accuracy, feed_dict = {X: shuff_X_val[:100], y: shuff_y_val[:100], keep_prob: 1.})

            #print the accuracy
            sys.stdout.write("\r" + "Epoch: " + str(epoch) + " ||| Validation Accuracy: " + str(validation_accuracy))
            sys.stdout.flush()

        saver.save(sess, save_file)
        print("")
        print("Trained Model Saved.")

        #check the final testing accuracy
        shuff_X_test, shuff_y_test = shuffle(xtest, ytest)
        testing_accuracy = sess.run(accuracy, feed_dict = {X: shuff_X_test[:200], y: shuff_y_test, keep_prob: 1.})
        print("")
        print("Testing Accuracy:", testing_accuracy)

        ender = timer()
        print("")
        print("Time:", ender - starter)

#train(model, epochs, batch_size, keep_prob_percent, X_train, y_train, X_val, y_val, X_test, y_test, save_model()[0], save_model()[1])

tf.reset_default_graph

def test_accuracy():
    with tf.Session(graph = model) as sess:
        save_model()[0].restore(sess, save_model()[1])
        feed_dict = {X: X_test[:200], y: X_test[:200], keep_prob: 1.}

        file_writer = tf.summary.FileWriter('./logs/model_graph', sess.graph)
        print("Test Accuracy:", accuracy.eval(feed_dict = feed_dict))

    tf.reset_default_graph()

#test_accuracy()
