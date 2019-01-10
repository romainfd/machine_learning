'''
Feature extractor to feed neural network

Converts the training set and the testing set into a new pair of training set/testing set with the features computed.
'''

import random
import numpy as np
from scipy import sparse

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import nltk
import csv
import pickle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

# creates
nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

with open("input/testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]


""" We define here the relative importance of the different parts. """
title = "clean"  # the name used for the computed files
FEATURES_SIZE = 0.7  # The size of the dataset used to build the graphs (authors and citations graphs used to compute the features)
TRAINING_SIZE = 0.3  # The size of the dataset used to build the training features (they are later spli into training and validation in the ML algorithms)
USED = FEATURES_SIZE + TRAINING_SIZE
assert USED <= 1  # We make sure we are asking for something relevant



######################################################################################################################
'''
Data loading and preprocessing 


The columns of the data frame below are: 
(1) paper unique ID (integer)
(2) publication year (integer)
(3) paper title (string)
(4) authors (strings separated by ,)
(5) name of journal (optional) (string)
(6) abstract (string) - lowercased, free of punctuation except intra-word dashes
'''

### 1. We collect and split the data

# 1.a import training set
with open("input/training_set.txt", "r") as f:
    reader = csv.reader(f)
    dataset = list(reader)
dataset = [element[0].split(" ") for element in dataset]

# 1.b convert labels into integers then into column array
labels = [int(element[2]) for element in dataset]
labels = list(labels)
labels_array = np.array(labels)

# 1.c Split the data
# Using Skicit-learn to split data into fata used for features and training set
from sklearn.model_selection import train_test_split

# Split the data into other and features sets
other_set, features_set, other_labels_array, features_labels_array = train_test_split(dataset, labels_array, test_size=FEATURES_SIZE, random_state=42)
# Split the other data into training and not used sets
training_part = min(1., TRAINING_SIZE / (1 - FEATURES_SIZE))  # to make sure to have the exact value we asked for, independantly of the FEATURES_SIZE
training_set, _, labels_array, _ = train_test_split(other_set, other_labels_array, test_size=(1 - training_part), random_state=42)

print("DATA SEPARATION DONE")

### 2. We build the graphs used to compute the features on the features_set

# 2.a We load the articles information
with open("input/node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)

IDs = [element[0] for element in node_info]


# 2.b We compute the TF-IDF vector of each paper
corpus = [element[5] for element in node_info]  # our vocabulary
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

corpus_tokenized = [par.lower().split(" ") for par in corpus]
corpus_tokenized = [[token for token in par if token not in stpwds] for par in corpus_tokenized]

# 2.c We also build a Doc2Vec representation
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus_tokenized)]
model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
Doc2Vec_representation = [ model.infer_vector(par) for par in corpus_tokenized ]


#######################################################################################
# 2.d We build the dictionaries for later use and quick access to data
authors = [element[3].split(", ") for element in node_info]

author_index = {}  # dictionary to give authors unique IDs
paper_index = {}  # dictionary to link paper IDs to their index in node_info

# populate author dictionary
for row in authors:
    for name in row:
        if name != "" and not(name in author_index):
            author_index[name] = len(author_index)
author_index[''] = -1

# populate paper dictionary
for i in range(len(IDs)):
    paper_index[IDs[i]] = i

N = len(author_index)

# 2.e We build the adjacency matrix of the authors graph
# adjacency matrix for the author based graph
A = np.zeros((N, N), dtype = np.int)

features_set_true = [element for element in features_set if element[2]]

for edge in features_set_true:
    source_id = paper_index[str(edge[0])]
    target_id = paper_index[str(edge[1])]
    for source_author in authors[source_id]:
        if source_author == "":  # check if there is an author from the source paper
            continue
        for target_author in authors[target_id]:
            if target_author == "":  # check if there is an author from the target paper
                continue
            A[author_index[source_author], author_index[target_author]] += 1

A = sparse.csr_matrix(A)

#
degree_out_a = [ np.sum(A[i]) for i in range(N) ]
degree_in_a = [ np.sum(A[:,i]) for i in range(N) ]

# 2-length path
AA = np.dot(A, A)
AAt = np.dot(A, A.T)
AtA = np.dot(A.T, A)

# 3-length path
AAA = np.dot(AA, A)
AtAA = np.dot(AtA, A)
AAtA = np.dot(AAt, A)
AAAt = np.dot(A, AAt)



print("AUTHORS GRAPHS COMPUTATIONS DONE")

##########################################################################################
# 2.f We build the adjacency matrix of the citations graph
N = len(IDs)
D = np.zeros((N,N))
for line in features_set:  ## WE USE ONLY FEATURES SET
    if int(line[2])==1:
        i = IDs.index(line[0]) # index_source
        j = IDs.index(line[1]) # index_target
        D[i,j] = 1

D = sparse.csr_matrix(D)
# 
degree_out = [ np.sum(D[i]) for i in range(N) ] # N = len(IDs)
degree_in = [ np.sum(D[:,i]) for i in range(N) ]

# 2-length path
DD = np.dot(D, D)
DDt = np.dot(D, D.T)
DtD = np.dot(D.T, D)

# 3-length path
DDD = np.dot(DD, D)
DtDD = np.dot(DtD, D)
DDtD = np.dot(DDt, D)
DDDt = np.dot(D, DDt)

print("PAPER GRAPHS COMPUTATIONS DONE")



## 3. FEATURES EXTRACTION: for training set now
# for each training example we need to compute its features
# in this baseline we will train the model on only TRAINING_SIZE % of the data set
def computeFeatures(training_set):
    f_degree_in_source = []
    f_degree_in_target = []
    f_degree_out_source = []
    f_degree_out_target = []

    f_DD = []
    f_DDt = []
    f_DtD = []

    f_DDD = []
    f_DtDD = []
    f_DDtD = []
    f_DDDt = []

    a_degree_in_source = []
    a_degree_in_target = []
    a_degree_out_source = []
    a_degree_out_target = []

    f_AA = []
    f_AAt = []
    f_AtA = []

    f_AAA = []
    f_AtAA = []
    f_AAtA = []
    f_AAAt = []


    # we will use three basic features:

    # number of overlapping words in title
    overlap_title = []

    # temporal distance between the papers
    temp_diff = []

    # number of common authors
    comm_auth = []

    # cosine similarity between both source and target tf-idf vectors
    TFIDF_cosine_similarity = []

    # Doc2Vec cosine similarity
    Doc2Vec_cosine_similarity = []

    # Author proximity
    graph_author_feature = []

    # titles cited in abstract
    title_cited = []

    # journals overlap
    journals = []

    for i in tqdm(range(len(training_set))):
        f = 0

        # 1. We get the 2 papers IDs
        source_id = paper_index[str(training_set[i][0])]
        target_id = paper_index[str(training_set[i][1])]


        degree_in_source = 0
        degree_in_target = 0
        degree_out_source = 0
        degree_out_target = 0
        faa = 0
        faat = 0
        fata = 0
        faaa = 0
        fataa = 0
        faata = 0
        faaat = 0

        # 2. We compute authors related features
        for source_author in authors[source_id]:
            if source_author == "":  # check if there is an author from the source paper
                continue
            for target_author in authors[target_id]:
                if target_author == "":  # check if there is an author from the target paper
                    continue
                f += A[author_index[source_author], author_index[target_author]]
                degree_in_source += degree_in_a[author_index[source_author]]
                degree_in_target_temp = degree_in_a[author_index[target_author]]
                degree_in_target += degree_in_target_temp
                degree_out_source_temp = degree_out_a[author_index[source_author]]
                degree_out_source += degree_out_source_temp
                degree_out_target += degree_out_a[author_index[target_author]]
                faa += AA[author_index[source_author], author_index[target_author]]
                faat += AAt[author_index[source_author], author_index[target_author]]
                fata += AtA[author_index[source_author], author_index[target_author]]
                faaa += AAA[author_index[source_author], author_index[target_author]]
                fataa += AtAA[author_index[source_author], author_index[target_author]]
                faata += AAtA[author_index[source_author], author_index[target_author]]
                faaat += AAAt[author_index[source_author], author_index[target_author]]

        # 3. We add the computed features
        a_degree_in_source.append(degree_in_source)
        a_degree_in_target.append(degree_in_target)
        a_degree_out_source.append(degree_out_source)
        a_degree_out_target.append(degree_out_target)

        f_AA.append(faa)
        f_AAt.append(faat)
        f_AtA.append(fata)

        f_AAA.append(faaa)
        f_AtAA.append(fataa)

        f_AAtA.append(faata)
        f_AAAt.append(faaat)

        graph_author_feature.append(f)


        # 4. We collect source/target info and clean it with our corpus
        source = training_set[i][0]
        target = training_set[i][1]

        index_source = IDs.index(source)
        index_target = IDs.index(target)

        source_info = [element for element in node_info if element[0]==source][0]
        target_info = [element for element in node_info if element[0]==target][0]

        # convert to lowercase and tokenize
        source_title = source_info[2].lower().split(" ")
        # remove stopwords
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]

        target_title = target_info[2].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]

        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")

        # 5. We compute easy features
        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
        TFIDF_cosine_similarity.append(float(cosine_similarity(features_TFIDF[index_source,:], features_TFIDF[index_target,:])))
        Doc2Vec_cosine_similarity.append(float(cosine_similarity(Doc2Vec_representation[index_source].reshape(1,-1), Doc2Vec_representation[index_target].reshape(1,-1))))

        # Is the dest_title in the source_abstract
        title_cited.append(1 if target_info[2].lower() in source_info[5] else 0)

        # Are the journals related
        jd = set(target_info[4].split('.'))
        if '' in jd:
            jd.remove('')
        js = set(source_info[4].split('.'))
        if '' in js:
            js.remove('')
        inter_len = len(js.intersection(jd))
        min_len = min(len(js), len(jd))
        journals.append(inter_len if min_len == inter_len else 0)

        # 6. We compute and add features based on the authors graphs
        f_degree_in_source.append(degree_in[index_source])
        f_degree_in_target.append(degree_in[index_target])
        f_degree_out_source.append(degree_out[index_source])
        f_degree_out_target.append(degree_out[index_target])

        f_DD.append(DD[index_source, index_target])
        f_DDt.append(DDt[index_source, index_target])
        f_DtD.append(DtD[index_source, index_target])

        f_DDD.append(DDD[index_source, index_target])
        f_DtDD.append(DtDD[index_source, index_target])

        f_DDtD.append(DDtD[index_source, index_target])
        f_DDDt.append(DDDt[index_source, index_target])

    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    training_features = np.array([overlap_title, temp_diff, comm_auth, TFIDF_cosine_similarity, Doc2Vec_cosine_similarity, f_degree_in_source, f_degree_in_target, f_degree_out_source, f_degree_out_target, f_DD, f_DDt, f_DtD, f_DDD, f_DtDD, f_DDtD, f_DDDt, graph_author_feature, a_degree_in_source, a_degree_in_target, a_degree_out_source, a_degree_out_target, f_AA, f_AAt, f_AtA, f_AAA, f_AtAA, f_AAtA, f_AAAt, title_cited, journals]).T

    # scale
    training_features = preprocessing.scale(training_features)

    return training_features

# Uncomment if we want to store the features of the samples used to build the graphs
# pickle.dump(computeFeatures(features_set), open('features/features_features_'+title+'.pkl', 'wb'))
# pickle.dump(features_labels_array, open('features/features_labels_'+title+'.pkl', 'wb'))

# We store our training features and labels
pickle.dump(computeFeatures(training_set), open('features/training_features_'+title+'.pkl', 'wb'))
pickle.dump(labels_array, open('features/training_labels_'+title+'.pkl', 'wb'))

# We store our testing features
pickle.dump(computeFeatures(testing_set), open('features/testing_features_'+title+'.pkl', 'wb'))
