## Data integreation
Add the following data files in the input folder:
* training_set.txt
* test_set.txt
* node_information.csv

## Final commands
### 1. Dependencies
To install the dependencies: `make dep`

### 2. Data Exploration and Random Forest baseline
We explored the features in a notebook.
To run the exploration notebook: `make explo`
We analyse the impact of the presence of the target title or authors in the source abstract, of the journals and of the dates.
Then, we quickly prepare the data to make a first random forest as baseline: tf-idf on titles, authors overlap, date difference.
We get around 78% of accuracy.
To learn more about our features we display features importance and some decision trees.

### 3. Features extraction
`make features` will launch the features extraction.
We build a graph of authors and a graph of papers on a part of the data (70%).
Then, we use this graphs to compute features on our links for the rest of the data and the test dataset.

### 4. First attempts
We built a SVM (`make svm`), an autoencoder (`make ae`) and a convolutionnal network (`make cnn`) but they had poor results.

### 5. Improved Random Forests
Using the previously computed features, we made a random exploration of parameters for a Random Forest.
The analysis of the features importance helped us better design our features and our NN.
The best model we found had these parameters: '50, sqrt, 25, 10, 16, False'
We reached a training error of 97.34% and a validation error of 96.92%.
On kaggle, this model reached 96.77% on the public dataset and 97.1% on the private one.

Some results are available [here](images).

### 6. Neural network
`make nn` to run our optimized Neural Network.



## Old commands
To launch the baseline algorithm:
`make base`

To send some results to the kaggle leaderboard
* install Kaggle
* Go to Kaggle > MY Profile and download the kaggle.json credentials to put on your computer (should be in ~/.kaggle)
* `FILE=file_name_without_.csv MESS="message" make send`
    * ex : `FILE=improved_predictions MESS="Initial test" make send`
   