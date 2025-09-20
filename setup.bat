@echo off
REM Simple setup script for Twitter Analysis

echo Installing dependencies...
pip install -r requirements.txt

echo Downloading spaCy model...
python -m spacy download en_core_web_sm

echo Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

echo Setup complete! Run 'python src/main.py --help' to get started.
pause
