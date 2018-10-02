import numpy as np
from numpy.linalg import norm,svd
from time import time
from sys import stdout


def nmf_factor(V, r, iterations=100):
    n, m = V.shape
    # W = np.ones((n, r))
    # H = np.ones((r, m))
    d_iter = np.ones(iterations)

    # we find the initial values of W and H using SVD and absolute value
    w0, eigvals, h0 = svd(V)
    W = abs(w0[:, :r])
    H = abs(h0[:r, :])
    for i in range(iterations):
        # we update H
        normalisation = 1. * np.transpose(W).dot(W).dot(H)
        num = np.transpose(W).dot(V)
        for a in range(r):
            for j in range(m):
                H[a, j] *= num[a, j] / normalisation[a, j]
        # we update W using the new value of H
        normalisation = 1. * W.dot(H).dot(np.transpose(H))
        num = V.dot(np.transpose(H))
        for j in range(n):
            for a in range(r):
                W[j, a] *= num[j, a] / normalisation[j, a]
        # we compute the distance
        d_iter[i] = norm(V - W.dot(H), ord='fro')
    return W, H, d_iter
