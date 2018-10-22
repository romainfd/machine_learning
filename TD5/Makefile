all: dep cnn_text_categorization.ipynb data
	jupyter notebook cnn_text_categorization.ipynb

dep: requirements.txt
	pip install	-r requirements.txt
	ipython kernel install --user --name=TD5
