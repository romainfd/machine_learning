import numpy as np

def linearKernel(X1, X2):
    # Computes the linear Kernel between two set of features
    m = X1.shape[0]
    n = X2.shape[0]
    K = np.zeros((m, n))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the linear kernel (see the assignment
    #				for more details).
    for i in range(m):
        for j in range(n):
            K[i, j] = np.sum(np.cross(X1[i], X2[j]))
    # =============================================================
        
    return K
