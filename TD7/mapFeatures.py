import numpy as np

def mapFeatures(X, degree=6):
    '''
    Generate a new feature matrix consisting of all polynomial combinations of 
    the features with degree less than or equal to the specified degree. 
    '''
    map = []

    # TODO
    for elem in range(X.shape[0]):
        entry = []
        for i in range(degree):
            for j in range(degree - i + 1):
                entry.append(X[elem, 0]**i * X[elem, 1]**j)
        map.append(entry)

    return np.array(map)
