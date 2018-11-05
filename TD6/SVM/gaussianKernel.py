import numpy as np

def gaussianKernel(X1, X2, sigma = 0.1):
    m = X1.shape[0]
    n = X2.shape[0]
    K = np.zeros((m, n))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the Gaussian kernel (see the assignment
    #				for more details).
    for i in range(m):
        for j in range(n):
            norm = np.sum((X1[i] - X2[j]) ** 2)
            K[i, j] = np.exp(- norm / (2 * (sigma ** 2)))
    # =============================================================

    return K
