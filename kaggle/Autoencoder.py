"""
Autoencoder based model using features computed previously:
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

First train an autoencoder model on only one kind of samples (label 0)
Compute the error of reconstruction on each feature and use it as input for the model

Then train a multi layer perceptron with all the samplesin order to classify based on the error by features

"""

#####################################################################################
'''
import of libraries
'''

import pickle
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import csv

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras import regularizers



#####################################################################################
'''
import features computed previously
'''

training_features = pickle.load(open("features/training_features_split.pkl",'rb'))
training_labels = pickle.load(open("features/training_labels_split.pkl",'rb'))
testing_features = pickle.load(open("features/testing_features_split.pkl",'rb'))



#####################################################################################
'''
Definition of the autoencoder
'''

input_dim = training_features.shape[1]

input_layer = Input(shape=(input_dim,))
hidden_layer = Dense(16, activation='tanh')(Dropout(0.)(input_layer))
encoded = Dense(6, activation='tanh')(hidden_layer)
decoded = Dense(input_dim, activation='tanh')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(loss='mean_squared_error', optimizer='RMSProp')

#####################################################################################
'''
autoencoder training
'''

# select only the negative labels to train the autoencoder
ae_training = training_features[training_labels==0]

history_ae = autoencoder.fit(ae_training, ae_training, validation_split=0.2, epochs=20, batch_size=64, verbose=2, shuffle=True)

# vizualisation of the loss during training
plt.plot(history_ae.history['loss'])
plt.plot(history_ae.history['val_loss'])
plt.title('autoencoder loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()



#####################################################################################
'''
Classifier based on the output of the autoencoder
'''

x1 = Input(shape=(input_dim,))
x2 = Input(shape=(input_dim,))
tmp = keras.layers.Subtract()([x1, x2])
mse = keras.layers.Multiply()([tmp, tmp])
output = Dense(1, activation='sigmoid')(mse)
model = Model([x1,x2], output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])



#####################################################################################
'''
classifier training
'''

history = model.fit([training_features, autoencoder.predict(training_features)], training_labels, validation_split=0.2, epochs=500, batch_size=512, verbose=2, shuffle=True)

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



#####################################################################################
'''
Saves the predictions of this model
'''

predictions = ((model.predict([testing_features,autoencoder.predict(testing_features)])>0.5)*1).reshape(-1)

predictions = zip(range(len(testing_features)), predictions)
with open("predictions/Autoencoder_predictions.csv","w") as pred1:
    csv_out = csv.writer(pred1)
    csv_out.writerow(["id", "category"])
    for row in predictions:
        csv_out.writerow(row)

