"""
SVM classifier using features computed previously:

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

#####################################################################################
'''
import libraries
'''

import pickle
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import csv
from sklearn import svm



#####################################################################################
'''
import features computed previously
'''

training_features = pickle.load(open("features/training_features_split.pkl",'rb'))
training_labels = pickle.load(open("features/training_labels_split.pkl",'rb'))
testing_features = pickle.load(open("features/testing_features_split.pkl",'rb'))


from sklearn.model_selection import train_test_split

train_features, validation_features, train_labels, validation_labels = train_test_split(training_features, training_labels, test_size = 0.2, random_state = 42)


#####################################################################################
'''
SVM Classifier
'''

# training
classifier = svm.SVC(kernel='rbf', verbose=3)
classifier.fit(training_features, training_labels)

# prediction
pred = classifier.predict(validation_features)
accuracy = 1-np.mean(np.abs(pred - validation_labels))
print('accuracy : {}'.format(accuracy))

#####################################################################################
'''
saves the predictions of the svm classifier
'''

predictions_SVM = list(classifier.predict(testing_features))
predictions_SVM = zip(range(len(testing_features)), predictions_SVM)

with open("predictions/SVM_predictions.csv","w") as pred1:
    csv_out = csv.writer(pred1)
    csv_out.writerow(["id", "category"])
    for row in predictions_SVM:
        csv_out.writerow(row)

