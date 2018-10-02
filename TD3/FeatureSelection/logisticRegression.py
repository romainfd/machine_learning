# Implements logistic Regression

from numpy import *

def logisticRegression(X,Y):
    # estimate w_logistic
    n_dimensions = X.shape[0]
    n_data = X.shape[1]
    # difference between prediction and real vlue
    dif = 5
    # weights on the features
    w = zeros((n_dimensions,1))
    # X' columns are the features
    Xt = transpose(X)

    # Logistic regression
    for i in range(20):
        Yn = 1/(1+exp(-dot(Xt,w)))

        #jacobian
        J = dot(X,(Y-Yn))
        #hessian
        R = zeros((n_data,n_data))
        for j in range(n_data):
            R[j,j] = Yn[j]*(1-Yn[j])
        
        H = dot(-X,dot(R,Xt))
    
        wn = w - linalg.lstsq(H,J)[0]  
    
        dif = sum(abs(Y-Yn))/X.shape[0]
        #print (dif)
        if dif <= 0.001:
            print ('here')

            break
        
        w = wn
    
    print
    return w
