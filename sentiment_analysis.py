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
    
neg_words_common, pos_words_common, neg_words_uncommon, pos_words_uncommon = create_common_uncommon_words(neg_list, pos_list, range(50), range(15000))

# %%
print(pos_words_common)
print("")
print(neg_words_common)

# %%
#remove the 60 most common words from all the reviews

for negative_review in range(len(neg_list)):
    negative_review_words = neg_list[negative_review].split()
    for words_in_review in range(len(negative_review_words)):
        if negative_review_words[words_in_review] in neg_words_common:
            negative_review_words[words_in_review] = ""
    neg_list[negative_review] = " ".join(negative_review_words)
    neg_list[negative_review] = " ".join(neg_list[negative_review].split())
    
for positive_review in range(len(pos_list)):
    positive_review_words = pos_list[positive_review].split()
    for words_in_review in range(len(positive_review_words)):
        if positive_review_words[words_in_review] in pos_words_common:
            positive_review_words[words_in_review] = ""
    pos_list[positive_review] = " ".join(positive_review_words)
    pos_list[positive_review] = " ".join(pos_list[positive_review].split())
    
# %%
print(neg_list[0])
print("")
print(pos_list[0])