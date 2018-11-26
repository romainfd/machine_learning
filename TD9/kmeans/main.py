from numpy import *
from sklearn import datasets
from kmeans import kmeans
from read_dataset import read_dataset
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


##############################################
## PART A: Artificial data
##############################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
data_art, labels_art = make_blobs(n_samples=750, centers=centers, cluster_std=0.3,
                            random_state=0)

# Visualize dataset
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.scatter(data_art[:,0],data_art[:,1], c=labels_art)
ax1.set_xlabel('1st dimension')
ax1.set_ylabel('2nd dimension')
ax1.set_title("Vizualization of the dataset")
plt.show()

# Run k-means algorithm for different values of k
k = 4
labels_pred_art = kmeans(data_art,k)


# Plot clustering results
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.scatter(data_art[:,0],data_art[:,1], c=labels_pred_art)
ax2.set_xlabel('1st dimension')
ax2.set_ylabel('2nd dimension')
ax2.set_title("Vizualization of the clusters produced by k-means algorithm")
plt.show()


##############################################
## PART B: MNIST dataset
##############################################
# Number of instaces and number of principal components (features)
n_instances = 1000
pca_features = 8

# Get the labels of each digit
images, labels_mnist = read_dataset(n_instances, pca_features);

# Create the dataset (data_mnist) that will be used in clustering
# load the PCA features of the test data set
data_mnist = loadtxt("test_data.csv",delimiter=',')
data_mnist = data_mnist[:n_instances,:pca_features] #only 8 first features are kept 

# Plot 2 out of 8 dimensions of the dataset - colors correspond to true labels
# (Hint: experiment with different combinations of dimensions)
# Only for illustration purposes
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.scatter(data_mnist[:,0],data_mnist[:,1], c=labels_mnist)
ax3.set_xlabel('1st dimension')
ax3.set_ylabel('2nd dimension')
ax3.set_title("Vizualization of the dataset (2D)")
plt.show()

# Run k-means algorithm for different values of k
k = 10
labels_pred_mnist = kmeans(data_mnist,k)

# Plot clustering results
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
ax4.scatter(data_mnist[:,0],data_mnist[:,1], c=labels_pred_mnist)
ax4.set_xlabel('1st dimension')
ax4.set_ylabel('2nd dimension')
ax4.set_title("Vizualization of the clusters produced by k-means algorithm in 2D space")
plt.show()

