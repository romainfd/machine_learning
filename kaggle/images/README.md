Here are the images output

* tree.png is the display of a Decision Tree built with:
  * all the features
  * a max-depth of 4
  * at least 16 samples in the leaves
  * no bootstrap of the data
  - This Decision Tree scored 95.62% on the testing set and 95.70% on the training set.

* features.png is the display of the features importance. It's based on a Random Forest with:
  * 50 Decision Trees
  * 3 features each
  * a max-depth of 20
  * at least 20 samples to split
  * at least 2 samples in a leaf
  * no bootstrap of the data
  - This Random Forest achieved our best score on the testing set: 97.02% (with 99.14% on the testing set).