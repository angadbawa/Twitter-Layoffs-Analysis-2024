.PHONY: help install install-dev test lint format clean docker-build docker-run docker-stop

help:
	@echo "Available commands:"
	@echo "  install      - Install package and dependencies"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-stop  - Stop Docker containers"
	@echo "  notebook     - Start Jupyter Lab"
	@echo "  analyze      - Run full analysis pipeline"

# Installation
install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,jupyter,viz]"
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Testing
test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=src/twitter_analysis --cov-report=html

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Development
notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Analysis
analyze:
	python src/main.py --mode full --num-tweets 500

analyze-sample:
	python src/main.py --mode full --num-tweets 100

collect-data:
	python src/main.py --mode collect --num-tweets 1000

preprocess-data:
	python src/main.py --mode preprocess --input-file raw_tweets.csv

# Setup
setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate     (Windows)"

# Documentation
docs:
	@echo "Documentation available in README.md"
	@echo "API documentation can be generated with sphinx"

# Health check
check:
	python -c "import src.twitter_analysis; print('✅ Package imports successfully')"
	python -c "from transformers import pipeline; print('✅ Transformers available')"
	python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy model available')"
