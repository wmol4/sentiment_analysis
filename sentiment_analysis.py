# %%
# Import packages
#keep adding packages as needed and re-running this cell
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import glob

# %%
#import data
#2 columns: data, labels
#load in the text files
#remove any symbol ( ,  . ! ? )
neg_review = glob.glob("W:/Projects/Sentiment Analysis/sentiment_analysis/review_polarity/txt_sentoken/neg/*.txt")
pos_review = glob.glob("W:/Projects/Sentiment Analysis/sentiment_analysis/review_polarity/txt_sentoken/pos/*.txt")

neg_list = []
pos_list = []
symbols = """-0123456789/\!@#$?"'():;,."""

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

# %%
#remove
#print(neg_list[0])
test_list = ['hello', 'hello man', 'hey this', 'man']
from collections import Counter
print(test_list.)
