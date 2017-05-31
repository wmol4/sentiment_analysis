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
neg_string = ""
for i in range(1000):
    neg_string = neg_string + " " + neg_list[i]
total_neg = neg_string.split()

pos_string = ""
for i in range(1000):
    pos_string = pos_string + " " + pos_list[i]
total_pos = pos_string.split()

neg_count = Counter(total_neg)
neg_words = []
neg_words_list = neg_count.most_common(60)
for i in range(len(neg_words_list)):
    neg_words.append(neg_words_list[i][0])

pos_count = Counter(total_pos)
pos_words = []
pos_words_list = pos_count.most_common(60)
for i in range(len(pos_words_list)):
    pos_words.append(pos_words_list[i][0])

print("")

# %%
#remove the 60 most common words from all the reviews

for negative_review in range(len(neg_list)):
    negative_review_words = neg_list[negative_review].split()
    for words_in_review in range(len(negative_review_words)):
        if negative_review_words[words_in_review] in neg_words:
            negative_review_words[words_in_review] = ""
    neg_list[negative_review] = " ".join(negative_review_words)
    neg_list[negative_review] = " ".join(neg_list[negative_review].split())
    
for positive_review in range(len(pos_list)):
    positive_review_words = pos_list[positive_review].split()
    for words_in_review in range(len(positive_review_words)):
        if positive_review_words[words_in_review] in pos_words:
            positive_review_words[words_in_review] = ""
    pos_list[positive_review] = " ".join(positive_review_words)
    pos_list[positive_review] = " ".join(pos_list[positive_review].split())
    
# %%
print(neg_list[0])
print("")
print(pos_list[0])