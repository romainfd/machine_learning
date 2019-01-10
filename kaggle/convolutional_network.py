"""
Convolutional Neural Network

Builds a convolutionnal network where the only inputs are the abstracts embedded as arrays where each row is a word embedded using word2vec. 
The idea was to train the network and extract the output of an intermediate layer which would then be used as an alternative to TF-iDF for representing the abstracts.
"""

import pickle
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import csv

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras import regularizers

from gensim.models.word2vec import Word2Vec

# Import abstracts
with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)

##################################################################################################
# Text embedding using word2vec


# parameters
compressed_length = 10
vector_size = 10
stpwds = set(nltk.corpus.stopwords.words("english"))

# building word2vec representation of the words
corpus = [element[5] for element in node_info]
corpus_tokenized = [par.lower().split(" ") for par in corpus]
corpus_tokenized = [[token for token in par if token not in stpwds] for par in corpus_tokenized]
model = Word2Vec(sentences=corpus_tokenized, size=vector_size, workers=2, sg=1, seed=0)
model.train(corpus_tokenized, total_examples=model.corpus_count, epochs=50)

# initialize abstract representation using word2vec embedding for the words
WV_abstract = []

# populate WV_abstract
for abstract in corpus_tokenized:
    vectors = np.zeros((compressed_length, vector_size))
    for i in range(compressed_length):
        word = abstract[(i*len(abstract))//compressed_length]
        if word in model.wv.vocab:
            vectors[i] = model.wv[word]
    WV_abstract.append(vectors)

# import training set
with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set = list(reader)

# prepare training set with abstracts represented as arrays according to WV_abstract
source_id = [int(element[0].split(" ")[0]) for element in training_set]
target_id = [int(element[0].split(" ")[1]) for element in training_set]
ID = [int(element[0]) for element in node_info]
labels = [int(element[0].split(" ")[2]) for element in training_set]

#initialize training set with 
training_source = np.zeros((len(training_set), 2*compressed_length, vector_size))

#populate training_source
for i in range(len(training_set)):
    source_index = ID.index(source_id[i])
    target_index = ID.index(target_id[i])
    training_source[i] = np.concatenate((WV_abstract[source_index], WV_abstract[target_index]), axis=0)

#reshape training source for CNN
training_source = training_source.reshape(len(training_source), 2*compressed_length, vector_size, 1)

#splitting training_source into training set and validation set
training_data, testing_data, training_labels, testing_labels = train_test_split(training_source, labels, train_size = 0.1, test_size=0.05, random_state = 16)
#################################################################################################


## Train model on training set

# build model
model = Sequential()
model.add(Conv2D(16, activation="relu", kernel_size=5, input_shape=(2*compressed_length, vector_size, 1), padding="valid"))
model.add(MaxPooling2D(5))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# train model
history = model.fit(training_data, training_labels, validation_data=(testing_data, testing_labels), verbose=2, epochs=50)


# plot learning curves

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
