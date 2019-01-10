"""
Neural Network using features computed previously:
    -'title'  # overlap between titles
    -'year'  # time difference
    -'authors'  # authors in common
    
    ## Abstract embedding
    -'tf-idf'  # tf-idf cosine similarity
    -'Doc2Vec'  # Doc2Vec cosine similarity
    
    ## Features computed on the connection between documents graph (represented by the adjacency matrix D)
    -'deg_in(s)'
    -'deg_in(t)'
    -'deg_out(s)'
    -'deg_out(t)'
    -'(DD)_st'
    -'(D D.T)_st'
    -'(D.T D)_st'
    -'(DDD)_st'
    -'(D.T DD)_st'
    -'(D D.T D)_st'
    -'(DD D.T)_st'
    
    ## Features computed on the connection between authors graph (represented by the adjacency matrix A)
    -'graph_author_feature'  # number of citations between both group of authors
    -'a_degree_in_source'  # sum of all citation received by the authors of the source document
    -'a_degree_in_target'  # sum of all citation received by the authors of the targe document
    -'a_degree_out_source'  # sum of all citation done by the authors of the source document
    -'a_degree_out_target'  # sum of all citation done by the authors of the target document
    -'f_AA'  # sum of the coefficient of the matrix AA over all position where i and j are respectively authors of the source and target documents
    -'f_AAt' # same with another matrix ...
    -'f_AtA'
    -'f_AAA'
    -'f_AtAA'
    -'f_AAtA'
    -'f_AAAt'
    
    ## Others
    -'title_cited'  # binary value if the title of the target is found in the abstract of the source
    -'journals'  # number of journals in common when the overlap is complete for one of the article

"""

######################################################################################
'''
import the libraries
'''

import pickle
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import csv
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers


######################################################################################
'''
import features computed previously
'''

# allows to keep only a subset of features
feature_list = ["title", "year", "authors", "tf-idf", "Doc2Vec", "deg_in(s)", "deg_in(t)", "deg_out(s)", "deg_out(t)", "(DD)_st", "(D D.T)_st", "(D.T D)_st", "(DDD)_st", "(D.T DD)_st", "(D D.T D)_st", "(DD D.T)_st", "graph_author_feature", "a_degree_in_source", "a_degree_in_target", "a_degree_out_source", "a_degree_out_target", "f_AA", "f_AAt", "f_AtA", "f_AAA", "f_AtAA", "f_AAtA", "f_AAAt"]
to_keep = feature_list  # change with the list of features wanted ["title", "year", "authors"] for example
to_keep_indices = [feature_list.index(feature) for feature in to_keep]

# import of the features and labels computed previously
features = pickle.load(open("features/features_features_split.pkl",'rb'))[:,to_keep_indices]
labels = np.array(pickle.load(open("features/features_labels_split.pkl",'rb')))
testing_features = pickle.load(open("features/testing_features_split.pkl",'rb'))[:,to_keep_indices]

# keep a validation set
n_cut = int(len(features) * 0.0714)
training_features = features[:n_cut]
training_labels = labels[:n_cut]

features = pickle.load(open("features/training_features_split.pkl",'rb'))[:,to_keep_indices]
labels = np.array(pickle.load(open("features/training_labels_split.pkl",'rb')))
n_cut = int(len(features) * 0.5)
validation_features = features[n_cut:]
validation_labels = labels[n_cut:]



######################################################################################
'''
Multi Layer Perceptron
'''

input_dim = training_features.shape[1]

## Define the model

model = Sequential()
model.add(Dense(512, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model.add(Dropout(0.))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])



######################################################################################
'''
Training
'''

history = model.fit(training_features, training_labels, validation_data=(validation_features, validation_labels), epochs=25, batch_size=5096, verbose=2, shuffle=True)


# plot learning curves

# accuracy during training
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# loss during training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()



######################################################################################
'''
saves the predictions of the MLP
'''

predictions = ((model.predict(testing_features)>0.5)*1).reshape(-1)

predictions = zip(range(len(testing_features)), predictions)

with open("predictions/NN_predictions.csv","w") as pred1:
    csv_out = csv.writer(pred1)
    csv_out.writerow(["id", "category"])
    for row in predictions:
        csv_out.writerow(row)

