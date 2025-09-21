# Twitter Layoffs Analysis 2024

This project involves the comprehensive analysis of Twitter data related to layoffs using various data science and natural language processing (NLP) techniques. The main objectives include data scraping, preprocessing, sentiment analysis, topic modeling, named entity recognition, keyword extraction, and emotion analysis.
  
## Key Steps

1. **Data Scraping:**
    - Extracted relevant Twitter data using ntscraper and nitter.

2. **Library Import:**
    - Imported essential Python libraries: pandas, numpy, re, matplotlib, gensim, spacy, sklearn, and transformers.

3. **Data Preprocessing and Cleaning:**
    - Understood data types, removed duplicates, and checked for null values.
    - Cleaned text data by removing mentions, emails, numbers, hashtags, links, emojis, unnecessary punctuation, and stopwords using regular expressions.

4. **Sentiment Analysis:**
    - Utilized the pretrained model "nlptown/bert-base-multilingual-uncased-sentiment" to analyze sentiment in the text data.
    - Plotted the distribution of sentiment scores.

5. **Topic Modeling:**
    - Applied the LDA (Latent Dirichlet Allocation) model to identify topics.
    - Displayed the top words for each topic and plotted topic distributions.
    - Used PCA for dimensionality reduction and K-means clustering to visualize topics in a scatter plot.

6. **Named Entity Recognition (NER):**
    - Employed the model 'en_core_web_sm' for NER to extract and plot the distribution of company names and key individuals mentioned.

7. **Keyword Extraction:**
    - Applied TFIDF Vectorizer to extract and score keywords.
    - Plotted the distribution of the top 20 keywords based on their TF-IDF scores.

8. **Emotion Analysis:**
    - Utilized the pretrained model "distilbert-base-uncased-finetuned-sst-2-english" to predict and plot the distribution of emotions in the text data.


### Setup
```bash
docker-compose up
```

```bash

Create Virtual Environment

python3 -m venv .env

Activate the virtual environment

.env\scripts\activate

Install the required dependencies

python3 -m pip install -r requirements.txt
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
