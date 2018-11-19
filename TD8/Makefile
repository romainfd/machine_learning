all: dep
	python main.py

test: dep
	python test_envs.py

dep: requirements.txt
	pip install --upgrade pip
	pip install -r requirements.txt --upgrade --timeout 300