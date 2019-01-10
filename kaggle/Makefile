base: public_baselines.py
	python public_baselines.py

explo: Data_Exploration_and_simple_RF.ipynb
	jupyter notebook Data_Exploration_and_simple_RF.ipynb

features: *
	python feature_extractor.py

rf: Random_Forest_improvements.ipynb
	jupyter notebook Random_Forest_improvements.ipynb

svm: SVM_classifier.py
	python SVM_classifier.py

ae: Autoencoder.py
	python Autoencoder.py

cnn: convolutional_network.py
	python convolutional_network.py

nn: neural_network.py
	python neural_network.py

dep: requirements.txt
	pip install --upgrade pip
	pip install -r requirements.txt --upgrade --timeout 300

send:
	kaggle competitions submit -c inf554-2018 -f predictions/$(FILE).csv -m "$(MESS)"