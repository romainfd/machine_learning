"""
Create a weighted oriented relational graph based on authors citing each other.
This graph is represented using its adjacency matrix.

On execution, will write a pkl file containing the adjacency matrix for authors
"""

import csv
import numpy as np
import pickle
import scipy.sparse

# import training and node information sets
with open("input/training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("input/node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)

IDs = [element[0] for element in node_info]  # list of node IDs in the order of node info
authors = [element[3].split(", ") for element in node_info] #list of list of authors for one paper in the order of node info

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

# adjacency matrix for the author based graph
author_adjacency = np.zeros((len(author_index),len(author_index)), dtype = np.int)

# training set reduced to entries where there is a citation
training_set_reduced = [element for element in training_set if element[2]]

# populate adjacency matrix
for edge in training_set_reduced:
    source_id = paper_index[str(edge[0])]
    target_id = paper_index[str(edge[1])]
    for source_author in authors[source_id]:
        if source_author == "":  # check if there is an author from the source paper
            continue
        for target_author in authors[target_id]:
            if target_author == "":  # check if there is an author from the target paper
                continue
            author_adjacency[author_index[source_author], author_index[target_author]] += 1

#################################################################################################################

# save all computed features
author_adjacency = scipy.sparse.csr_matrix(author_adjacency)
pickle.dump(author_adjacency, open('features/author_adjacency_1.pkl', 'wb'))

