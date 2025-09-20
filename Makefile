.PHONY: install test clean run notebook analyze

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('stopwords', 'punkt')"

test:
	python -m pytest tests/ -v

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/

run:
	docker-compose up -d

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

analyze:
	python src/main.py --mode full --num-tweets 500