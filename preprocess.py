# Preprocess the data using tokenization and stemming.

import nltk
import numpy as np
# nltk.download('all')

from nltk.stem import PorterStemmer

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def bagOfWords(tokenizedSentence, words):
    tokenizedSentence = [stem(w) for w in tokenizedSentence]
    bag = np.zeros(len(words))

    for i, w in enumerate(words):
        if w in tokenizedSentence:
            bag[i] = 1
    
    return bag