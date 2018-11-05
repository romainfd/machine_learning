# machine_learning
INF554 course @ École polytechnique about Machine Learning

# Use the code
_Remark: all the make command should be run in the folder of the project (see 'List of the projects' below)_
## Ready to use
### With the Makefile
Go in the TD project and run the `make` command you want (see 'List of the projects' below)

### With the python interpreter
* Run `python main.py`</br>
 _Remark: if some libraries are missing, you might need to run `make dep` to install them_
* To know more about the available flags use 'python main.py --help`
  * `python main.py --PCA=True` to PCA ([Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis))
  * `python main.py --degreeMax=7`
  * any combination of the two flags
  
## Modify the code
* Fork the repo by clicking on the fork button and then [clone your git repo on your computer](https://help.github.com/articles/cloning-a-repository/)
* Modify the code as desired
* Follow one of the two 'Ready to use' options above to execute your code

# List of the projects
1. Introduction to the Machine Learning Pipeline
* Pipeline explanation:
  * Loading and inspecting the provided data (temperature, soil moisture and number of new cells). 
  * Preprocessing to remove an outlier, normalize all the inputs and expand the feature space with polynomial basis functions to be able to fit with a linear model.
  * Use of a simple [Least Square regression](https://en.wikipedia.org/wiki/Linear_least_squares) to predict the number of new cells based on the temperature and soil moisture using linear combinations of polynomial functions. 
  * Illustration of overfitting for high degree polynomial functions.
* Pipeline in action:
  * `make all` to run
  * possibilities to add PCA ([Principal Component Analysis] or degreeMax flags

2. Use of 2 supervised learning methods
* `make kNN` to use the kNN method to recognized handwritten digits based on the [k-Nearest Neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* `make LR` to use [Logistic Regression method](https://en.wikipedia.org/wiki/Logistic_regression) to find the [decision boundary in a binary classification problem](https://en.wikipedia.org/wiki/Decision_boundary). Our use case is to determine if a student will be admitted to university based on his grades for 2 exams. To do so, we minimize a cost function (the errors on the prediction of admission) using a [batch gradient descent method](https://en.wikipedia.org/wiki/Gradient_descent) (from scipy or a mini-batch gradient descent I wrote but uncommenting it at the end of TD2/LR/main.py)

3. Various methods
* Feature Selection: `make FS`
* SVD: `make SVD`
* PCA: `make PCA`
* NMF: `make NMF`

4. Introduction to Tensorflow
* Neural Network to learn on the MNIST dataset
   * `make NN` to launch the model
* Neural Network to find the type of landscape based on characteristics of the picture
   * `make HW` to launch the model (and a 2-minute training)
   * The report is available [here](https://github.com/romainfd/machine_learning/blob/master/TD4/Report/Report.pdf)

5. Introduction to Keras
see the [Jupyter notebook](https://github.com/romainfd/machine_learning/blob/master/TD5/cnn_text_categorization.ipynb)

6. SVM and boosting on Decision Trees
* SVM:
   * First example: determining the decision boundary with a linear kernel
   * Second example: Gaussian kernel to find a [non-linear decision boundary](https://github.com/romainfd/machine_learning/blob/master/TD6/SVM/2.gaussian_sigma%3D0.05_C%3D50)
   * Third example: Gaussian kernel in case of blur boundary
   * `make SVM1` or `make SVM2` or `make SVM3`
* Adaboost:
   * Implementation of the [adaboost method on Decision Trees](https://github.com/romainfd/machine_learning/blob/master/TD6/Adaboost/Adaboost_100trees_depth%3D8) to reduce the variance
   * Evaluation of the [optimal depth](https://github.com/romainfd/machine_learning/blob/master/TD6/Adaboost/Adaboost_accuracy_vs_depth)
   * `make AB`
