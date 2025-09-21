# Twitter Layoffs Analysis 2024

Functional programming approach to analyzing Twitter data related to layoffs using NLP and ML techniques.

## Features

- **Data Scraping**: Twitter data collection using ntscraper
- **Text Processing**: Functional cleaning and preprocessing pipelines  
- **Sentiment Analysis**: BERT-based sentiment classification
- **Topic Modeling**: LDA-based topic discovery
- **NER**: Company and person extraction
- **Visualization**: Interactive dashboards and plots

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