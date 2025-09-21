from typing import List, Dict, Any, Callable, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import logging
from functools import partial
from ..utils.helpers import pipe, memoize
from ..utils.config import get_config


def create_tfidf_features(texts: List[str], max_features: int = 1000, 
                         min_df: int = 2, max_df: float = 0.8,
                         ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF features from text data.
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        ngram_range: Range of n-grams to extract
        
    Returns:
        Tuple of (feature matrix, fitted vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words='english'
    )
    
    features = vectorizer.fit_transform(texts)
    logging.info(f"Created TF-IDF features: {features.shape}")
    
    return features.toarray(), vectorizer


def create_count_features(texts: List[str], max_features: int = 1000,
                         min_df: int = 2, max_df: float = 0.8,
                         ngram_range: Tuple[int, int] = (1, 1)) -> Tuple[np.ndarray, CountVectorizer]:
    """
    Create count-based features from text data.
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        ngram_range: Range of n-grams to extract
        
    Returns:
        Tuple of (feature matrix, fitted vectorizer)
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words='english'
    )
    
    features = vectorizer.fit_transform(texts)
    logging.info(f"Created count features: {features.shape}")
    
    return features.toarray(), vectorizer


def extract_text_statistics(text: str) -> Dict[str, Any]:
    """
    Extract statistical features from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of text statistics
    """
    words = text.split()
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in text.split('.') if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'digit_count': sum(1 for c in text if c.isdigit()),
        'special_char_count': sum(1 for c in text if not c.isalnum() and not c.isspace())
    }


def add_text_features(df: pd.DataFrame, text_column: str = 'text_cleaned') -> pd.DataFrame:
    """
    Add text-based features to DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Column containing text to analyze
        
    Returns:
        DataFrame with additional text features
    """
    if df.empty or text_column not in df.columns:
        logging.warning(f"Column '{text_column}' not found or DataFrame is empty")
        return df
    
    df_enhanced = df.copy()
    
    text_stats = df_enhanced[text_column].apply(extract_text_statistics)
    stats_df = pd.DataFrame(text_stats.tolist())
    stats_df.columns = [f'text_{col}' for col in stats_df.columns]
    df_enhanced = pd.concat([df_enhanced, stats_df], axis=1)
    
    logging.info(f"Added {len(stats_df.columns)} text features")
    return df_enhanced


def extract_temporal_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Extract temporal features from date column.
    
    Args:
        df: Input DataFrame
        date_column: Column containing dates
        
    Returns:
        DataFrame with temporal features
    """
    if df.empty or date_column not in df.columns:
        logging.warning(f"Column '{date_column}' not found or DataFrame is empty")
        return df
    
    df_temporal = df.copy()
    
    # Convert to datetime if not already
    df_temporal[date_column] = pd.to_datetime(df_temporal[date_column], errors='coerce')
    
    # Extract temporal features
    df_temporal['year'] = df_temporal[date_column].dt.year
    df_temporal['month'] = df_temporal[date_column].dt.month
    df_temporal['day'] = df_temporal[date_column].dt.day
    df_temporal['hour'] = df_temporal[date_column].dt.hour
    df_temporal['day_of_week'] = df_temporal[date_column].dt.dayofweek
    df_temporal['is_weekend'] = df_temporal['day_of_week'].isin([5, 6]).astype(int)
    df_temporal['quarter'] = df_temporal[date_column].dt.quarter
    
    # Add time-based categories
    df_temporal['time_period'] = pd.cut(
        df_temporal['hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        include_lowest=True
    )
    
    logging.info("Added temporal features")
    return df_temporal


def encode_categorical_features(df: pd.DataFrame, categorical_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using label encoding.
    
    Args:
        df: Input DataFrame
        categorical_columns: List of categorical columns to encode
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    df_encoded = df.copy()
    encoders = {}
    
    for column in categorical_columns:
        if column in df_encoded.columns:
            encoder = LabelEncoder()
            df_encoded[f'{column}_encoded'] = encoder.fit_transform(df_encoded[column].astype(str))
            encoders[column] = encoder
            logging.info(f"Encoded categorical column: {column}")
    
    return df_encoded, encoders


def create_feature_matrix(df: pd.DataFrame, text_column: str = 'text_cleaned',
                         include_text_stats: bool = True,
                         include_temporal: bool = True,
                         vectorizer_type: str = 'tfidf') -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Create comprehensive feature matrix from DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Column containing cleaned text
        include_text_stats: Whether to include text statistics
        include_temporal: Whether to include temporal features
        vectorizer_type: Type of text vectorizer ('tfidf' or 'count')
        
    Returns:
        Tuple of (feature matrix, feature names, metadata)
    """
    if df.empty or text_column not in df.columns:
        logging.error("Cannot create feature matrix: DataFrame is empty or text column missing")
        return np.array([]), [], {}
    
    df_features = df.copy()
    feature_names = []
    metadata = {}

    if include_text_stats:
        df_features = add_text_features(df_features, text_column)
        text_stat_columns = [col for col in df_features.columns if col.startswith('text_')]
        feature_names.extend(text_stat_columns)
    
    if include_temporal and 'date' in df_features.columns:
        df_features = extract_temporal_features(df_features)
        temporal_columns = ['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'quarter']
        feature_names.extend([col for col in temporal_columns if col in df_features.columns])
    
    config = get_config("analysis")
    texts = df_features[text_column].tolist()
    
    if vectorizer_type == 'tfidf':
        text_features, vectorizer = create_tfidf_features(
            texts,
            max_features=config["max_features"],
            min_df=config["min_df"],
            max_df=config["max_df"]
        )
    else:
        text_features, vectorizer = create_count_features(
            texts,
            max_features=config["max_features"],
            min_df=config["min_df"],
            max_df=config["max_df"]
        )
    
    text_feature_names = [f'text_vec_{i}' for i in range(text_features.shape[1])]
    feature_names.extend(text_feature_names)

    other_features = df_features[feature_names[:-len(text_feature_names)]].values if feature_names[:-len(text_feature_names)] else np.array([]).reshape(len(df_features), 0)
    
    if other_features.size > 0:
        feature_matrix = np.hstack([other_features, text_features])
    else:
        feature_matrix = text_features
    
    metadata = {
        'vectorizer': vectorizer,
        'n_samples': len(df_features),
        'n_features': feature_matrix.shape[1],
        'text_feature_count': text_features.shape[1],
        'other_feature_count': other_features.shape[1] if other_features.size > 0 else 0
    }
    
    logging.info(f"Created feature matrix: {feature_matrix.shape}")
    return feature_matrix, feature_names, metadata


def normalize_features(features: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize feature matrix.
    
    Args:
        features: Feature matrix to normalize
        method: Normalization method ('standard', 'minmax', or 'robust')
        
    Returns:
        Tuple of (normalized features, normalization metadata)
    """
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    normalized_features = scaler.fit_transform(features)
    
    metadata = {
        'scaler': scaler,
        'method': method,
        'original_shape': features.shape,
        'normalized_shape': normalized_features.shape
    }
    
    logging.info(f"Normalized features using {method} scaling")
    return normalized_features, metadata


def create_sentiment_features(df: pd.DataFrame, text_column: str = 'text_cleaned') -> Tuple[np.ndarray, List[str]]:
    """Create features optimized for sentiment analysis."""
    features, names, _ = create_feature_matrix(
        df, text_column, 
        include_text_stats=True, 
        include_temporal=False,
        vectorizer_type='tfidf'
    )
    return features, names


def create_topic_modeling_features(df: pd.DataFrame, text_column: str = 'text_cleaned') -> Tuple[np.ndarray, List[str]]:
    """Create features optimized for topic modeling."""
    features, names, _ = create_feature_matrix(
        df, text_column,
        include_text_stats=False,
        include_temporal=False, 
        vectorizer_type='count'
    )
    return features, names

create_basic_tfidf = partial(create_tfidf_features, max_features=500, ngram_range=(1, 1))
create_advanced_tfidf = partial(create_tfidf_features, max_features=2000, ngram_range=(1, 3))
