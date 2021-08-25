import json
from preprocess import *
import numpy as np


with open('intent.json', 'r') as f:
    data = json.load(f)

words = []
intents = []
xy = []

# PREPROCESS (Tokenize + Stemming + Stop Word Removal)
for intent in data['intents']:
    tag = intent['intent']
    intents.append(tag)
    for text in intent['text']:
        w = tokenize(text)
        words.extend(w)
        xy.append((w, tag))

stop_words = ['?', ",", ".", "!", "a", "the"]

words = [stem(w) for w in words if w not in stop_words]
words = sorted(set(words))
intents = sorted(set(intents))

x_train = []
y_train = []

for (s, i) in xy:
    bag = bagOfWords(s, words)
    x_train.append(bag)

    tags = intents.index(i) # Have numbers for intents
    y_train.append(tags) 

x_train = np.array(x_train)
y_train = np.array(y_train)


