from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from functools import partial
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..utils.helpers import timer, memoize, safe_execute, chunk_list
from ..utils.config import get_config


@memoize
def load_sentiment_model(model_name: str = None) -> pipeline:
    """
    Load pre-trained sentiment analysis model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded sentiment analysis pipeline
    """
    if model_name is None:
        model_name = get_config("models.sentiment_model")
    
    try:
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            return_all_scores=True
        )
        
        logging.info(f"Loaded sentiment model: {model_name}")
        return sentiment_pipeline
    
    except Exception as e:
        logging.error(f"Failed to load sentiment model {model_name}: {e}")
        # Fallback to a simpler model
        try:
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device,
                return_all_scores=True
            )
            logging.info("Loaded fallback sentiment model")
            return sentiment_pipeline
        except Exception as e2:
            logging.error(f"Failed to load fallback model: {e2}")
            return None


def analyze_single_text_sentiment(text: str, model: pipeline) -> Dict[str, Any]:
    """
    Analyze sentiment of a single text.
    
    Args:
        text: Text to analyze
        model: Sentiment analysis pipeline
        
    Returns:
        Dictionary with sentiment scores
    """
    if not text or not text.strip():
        return {
            'label': 'NEUTRAL',
            'score': 0.0,
            'positive_score': 0.33,
            'negative_score': 0.33,
            'neutral_score': 0.34
        }
    
    try:
        # Truncate text if too long (BERT models have token limits)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        results = model(text)
        
        # Handle different model output formats
        if isinstance(results[0], list):
            scores = results[0]
        else:
            scores = results
        
        # Create standardized output
        sentiment_dict = {}
        for item in scores:
            label = item['label'].upper()
            score = item['score']
            
            # Map different label formats to standard ones
            if label in ['POSITIVE', 'POS', '1']:
                sentiment_dict['positive_score'] = score
            elif label in ['NEGATIVE', 'NEG', '0']:
                sentiment_dict['negative_score'] = score
            elif label in ['NEUTRAL', 'NEU', '2']:
                sentiment_dict['neutral_score'] = score
        
        # Find dominant sentiment
        if 'positive_score' in sentiment_dict and 'negative_score' in sentiment_dict:
            if sentiment_dict['positive_score'] > sentiment_dict['negative_score']:
                dominant_label = 'POSITIVE'
                dominant_score = sentiment_dict['positive_score']
            else:
                dominant_label = 'NEGATIVE'
                dominant_score = sentiment_dict['negative_score']
        else:
            # Fallback for single-class outputs
            dominant_label = scores[0]['label'].upper()
            dominant_score = scores[0]['score']
        
        # Ensure all scores are present
        sentiment_dict.setdefault('positive_score', 0.0)
        sentiment_dict.setdefault('negative_score', 0.0)
        sentiment_dict.setdefault('neutral_score', 0.0)
        
        sentiment_dict.update({
            'label': dominant_label,
            'score': dominant_score
        })
        
        return sentiment_dict
    
    except Exception as e:
        logging.warning(f"Sentiment analysis failed for text: {e}")
        return {
            'label': 'NEUTRAL',
            'score': 0.0,
            'positive_score': 0.33,
            'negative_score': 0.33,
            'neutral_score': 0.34
        }


@timer
def analyze_batch_sentiment(texts: List[str], model: pipeline, batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a batch of texts.
    
    Args:
        texts: List of texts to analyze
        model: Sentiment analysis pipeline
        batch_size: Size of processing batches
        
    Returns:
        List of sentiment analysis results
    """
    if not texts:
        return []
    
    results = []
    
    # Process in chunks to avoid memory issues
    for chunk in chunk_list(texts, batch_size):
        chunk_results = []
        
        for text in chunk:
            result = analyze_single_text_sentiment(text, model)
            chunk_results.append(result)
        
        results.extend(chunk_results)
        logging.info(f"Processed sentiment for {len(chunk)} texts")
    
    return results


@timer
def analyze_tweets_sentiment(df: pd.DataFrame, text_column: str = 'text_cleaned',
                           model_name: str = None, batch_size: int = 32) -> pd.DataFrame:
    """
    Analyze sentiment for tweets in a DataFrame.
    
    Args:
        df: DataFrame containing tweets
        text_column: Column containing text to analyze
        model_name: Name of sentiment model to use
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with sentiment analysis results
    """
    if df.empty or text_column not in df.columns:
        logging.warning(f"Column '{text_column}' not found or DataFrame is empty")
        return df
    
    # Load model
    model = load_sentiment_model(model_name)
    if model is None:
        logging.error("Failed to load sentiment model")
        return df
    
    df_sentiment = df.copy()
    
    # Analyze sentiment
    texts = df_sentiment[text_column].tolist()
    sentiment_results = analyze_batch_sentiment(texts, model, batch_size)
    
    # Add results to DataFrame
    sentiment_df = pd.DataFrame(sentiment_results)
    
    # Add prefix to avoid column conflicts
    sentiment_columns = {
        'label': 'sentiment_label',
        'score': 'sentiment_score',
        'positive_score': 'sentiment_positive',
        'negative_score': 'sentiment_negative',
        'neutral_score': 'sentiment_neutral'
    }
    
    sentiment_df = sentiment_df.rename(columns=sentiment_columns)
    
    # Concatenate with original DataFrame
    df_sentiment = pd.concat([df_sentiment, sentiment_df], axis=1)
    
    logging.info(f"Completed sentiment analysis for {len(df_sentiment)} tweets")
    return df_sentiment


def calculate_sentiment_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate sentiment statistics from analyzed DataFrame.
    
    Args:
        df: DataFrame with sentiment analysis results
        
    Returns:
        Dictionary with sentiment statistics
    """
    if df.empty or 'sentiment_label' not in df.columns:
        return {}
    
    stats = {
        'total_tweets': len(df),
        'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
        'sentiment_percentages': df['sentiment_label'].value_counts(normalize=True).to_dict(),
        'average_scores': {
            'positive': df['sentiment_positive'].mean() if 'sentiment_positive' in df.columns else 0,
            'negative': df['sentiment_negative'].mean() if 'sentiment_negative' in df.columns else 0,
            'neutral': df['sentiment_neutral'].mean() if 'sentiment_neutral' in df.columns else 0
        },
        'confidence_stats': {
            'mean_confidence': df['sentiment_score'].mean() if 'sentiment_score' in df.columns else 0,
            'median_confidence': df['sentiment_score'].median() if 'sentiment_score' in df.columns else 0,
            'std_confidence': df['sentiment_score'].std() if 'sentiment_score' in df.columns else 0
        }
    }
    
    return stats


def filter_by_sentiment(df: pd.DataFrame, sentiment: str, min_confidence: float = 0.5) -> pd.DataFrame:
    """
    Filter tweets by sentiment and confidence threshold.
    
    Args:
        df: DataFrame with sentiment analysis results
        sentiment: Sentiment to filter for ('POSITIVE', 'NEGATIVE', 'NEUTRAL')
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered DataFrame
    """
    if df.empty or 'sentiment_label' not in df.columns:
        return df
    
    mask = (df['sentiment_label'] == sentiment.upper())
    
    if 'sentiment_score' in df.columns:
        mask = mask & (df['sentiment_score'] >= min_confidence)
    
    filtered_df = df[mask].copy()
    logging.info(f"Filtered to {len(filtered_df)} tweets with {sentiment} sentiment (confidence >= {min_confidence})")
    
    return filtered_df


def get_sentiment_trends(df: pd.DataFrame, date_column: str = 'date', 
                        period: str = 'D') -> pd.DataFrame:
    """
    Calculate sentiment trends over time.
    
    Args:
        df: DataFrame with sentiment analysis and date columns
        date_column: Column containing dates
        period: Pandas period string ('D', 'W', 'M', etc.)
        
    Returns:
        DataFrame with sentiment trends
    """
    if df.empty or 'sentiment_label' not in df.columns or date_column not in df.columns:
        return pd.DataFrame()
    
    df_trends = df.copy()
    df_trends[date_column] = pd.to_datetime(df_trends[date_column])
    
    # Group by time period and sentiment
    trends = df_trends.groupby([
        pd.Grouper(key=date_column, freq=period),
        'sentiment_label'
    ]).size().unstack(fill_value=0)
    
    # Calculate percentages
    trends_pct = trends.div(trends.sum(axis=1), axis=0) * 100
    
    return trends_pct


# Specialized sentiment analysis functions
analyze_positive_sentiment = partial(filter_by_sentiment, sentiment='POSITIVE')
analyze_negative_sentiment = partial(filter_by_sentiment, sentiment='NEGATIVE')
analyze_neutral_sentiment = partial(filter_by_sentiment, sentiment='NEUTRAL')

# High confidence filters
filter_high_confidence_positive = partial(filter_by_sentiment, sentiment='POSITIVE', min_confidence=0.8)
filter_high_confidence_negative = partial(filter_by_sentiment, sentiment='NEGATIVE', min_confidence=0.8)
