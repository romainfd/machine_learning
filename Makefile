all: dep main.py data tools.py
	python main.py

dep: requirements.txt
	pip install --upgrade pip
	pip install -r requirements.txt --upgrade --timeout 300