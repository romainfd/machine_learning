from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Load the "gatlin" image data
X = loadtxt('SVD/gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the 
#               three factors in variables U, S and V
#
# compute the SVD decomposition
U, eigval, V = linalg.svd(X)

#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values 
#
X10 = U[:, :10].dot(diag(eigval[:10])).dot(V[:10, :])
X20 = U[:, :20].dot(diag(eigval[:20])).dot(V[:20, :])
X50 = U[:, :50].dot(diag(eigval[:50])).dot(V[:50, :])
X100 = U[:, :100].dot(diag(eigval[:100])).dot(V[:100, :])
X200 = U[:, :200].dot(diag(eigval[:200])).dot(V[:200, :])



#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
#
errors = []
normalisation = sqrt(sum(eigval[:]**2))
for k in [10, 20, 50, 100, 200]:
    err = sqrt(sum(eigval[k+1:]**2))
    errors.append(err / normalisation)

plt.figure(4)
plt.plot([10, 20, 50, 100, 200], errors)
plt.title('Frobenius error for each low rank approximation of the matrix')
plt.xlabel('Low rank k approximation (we take the top k eigenvalues)')
plt.ylabel('Frobenius error')
plt.draw()


#=========================================================================



# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X10,cmap = cm.Greys_r)
plt.title('Best rank' + str(10) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(X20,cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X50,cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X100,cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X200,cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k
plt.figure(3)
plt.plot(range(1, 1 + len(eigval)), eigval)
plt.title('Eigenvalue versus their rank')
plt.xlabel('Eigenvalue rank')
plt.ylabel('Eigenvalue')
plt.draw()


#=========================================================================

plt.show() 

