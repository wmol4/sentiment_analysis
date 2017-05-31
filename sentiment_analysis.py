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
#2 columns: data, labels
#load in the text files
#remove any symbol ( ,  . ! ? )
neg_review = glob.glob("W:/Projects/Sentiment Analysis/sentiment_analysis/review_polarity/txt_sentoken/neg/*.txt")
pos_review = glob.glob("W:/Projects/Sentiment Analysis/sentiment_analysis/review_polarity/txt_sentoken/pos/*.txt")

neg_list = []
pos_list = []
symbols = """~_-0123456789/\!@#$?"'():;,."""

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
#remove 50 most common words
neg_string = ""
for i in range(1000):
    neg_string = neg_string + " " + neg_list[i]
total_neg = neg_string.split()

pos_string = ""
for i in range(1000):
    pos_string = pos_string + " " + pos_list[i]
total_pos = pos_string.split()

neg_count = Counter(total_neg)
neg_words_list = neg_count.most_common(60)
pos_count = Counter(total_pos)
pos_words_list = pos_count.most_common(60)

print("")

# %%
test_string = "this is a test string and it is really cool"
words = ["test", "it"]
for word in words:
    test_string = test_string.replace(word, "")
print(test_string)
