# Feature selection with the Chi^2 measure

from numpy import *

def chiSQ(x, y):
    '''
        x: features (data)
        y: output (classes)
    '''
    cl = unique(y)  # unique number of classes
    rows = x.shape[0]
    dim = x.shape[1]
    chisq = zeros(dim)  # initialize array (vector) for the chi^2 values
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: calculate the importance for each feature

    for d in range(dim):
        feature = x[:, j]  # column j
        values = unique(feature)  # the possible values for feature j
        total = 0
        for i in range(len(values)):  # for each of the unique values of the feature
            v = values[i]
            # to finish
            for c_i in range(len(cl)):
                c = cl[c_i]
                Ovcj = 0
                Ixjv = 0
                Iyc = 0
                for i in range(rows):
                    Ovcj += (x[i][j] == v)*(y[i] == c)
                    Ixjv += (x[i][j] == v)
                    Iyc += (y[i] == c)
                Evcj = 1./rows * Ixjv * Iyc
                chisq[j] += pow(Ovcj - Evcj, 2)/Evcj



    return chisq
