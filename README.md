# Twitter Layoffs Analysis 2024

Functional programming approach to analyzing Twitter data related to layoffs using NLP and ML techniques.

## Features

- **Data Scraping**: Twitter data collection using ntscraper
- **Text Processing**: Functional cleaning and preprocessing pipelines  
- **Sentiment Analysis**: BERT-based sentiment classification
- **Topic Modeling**: LDA-based topic discovery
- **NER**: Company and person extraction
- **Visualization**: Interactive dashboards and plots

## Quick Setup

### Automated Setup
```bash
# Linux/Mac
chmod +x setup.sh && ./setup.sh

# Windows
setup.bat
```

### Manual Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Docker
```bash
docker-compose up
```

## Usage

```bash
# Run full analysis
python src/main.py --mode full --num-tweets 500

# Collect data only
python src/main.py --mode collect --num-tweets 1000

# Start Jupyter
jupyter lab
```

## Project Structure

```
src/
├── twitter_analysis/
│   ├── data/          # Data collection and loading
│   ├── preprocessing/ # Text cleaning and transformation  
│   ├── analysis/      # ML and NLP analysis
│   ├── visualization/ # Plotting and dashboards
│   └── utils/         # Helper functions
└── main.py           # CLI entry point
```

## Functional Programming Features

- Pure functions with no side effects
- Function composition with `compose()` and `pipe()`
- Immutable data transformations
- Higher-order functions and partial application
- Memoization for performance optimization