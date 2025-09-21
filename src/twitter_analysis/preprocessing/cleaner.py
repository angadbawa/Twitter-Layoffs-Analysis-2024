from typing import List, Callable, Optional
import re
import pandas as pd
import logging
from functools import partial
from ..utils.helpers import compose, pipe
from ..utils.config import get_config

PATTERNS = {
    'mentions': r'@\w+',
    'hashtags': r'#\w+',
    'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'emails': r'\S+@\S+',
    'numbers': r'\d+',
    'extra_whitespace': r'\s+',
    'punctuation': r'[^\w\s]',
    'emojis': r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
}


def remove_pattern(text: str, pattern: str, replacement: str = ' ') -> str:
    """
    Remove a regex pattern from text.
    
    Args:
        text: Input text
        pattern: Regex pattern to remove
        replacement: String to replace matches with
        
    Returns:
        Cleaned text
    """
    return re.sub(pattern, replacement, text)


def remove_mentions(text: str) -> str:
    """Remove Twitter mentions (@username) from text."""
    return remove_pattern(text, PATTERNS['mentions'])


def remove_hashtags(text: str) -> str:
    """Remove hashtags (#hashtag) from text."""
    return remove_pattern(text, PATTERNS['hashtags'])


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return remove_pattern(text, PATTERNS['urls'])


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return remove_pattern(text, PATTERNS['emails'])


def remove_numbers(text: str) -> str:
    """Remove numbers from text."""
    return remove_pattern(text, PATTERNS['numbers'])


def remove_emojis(text: str) -> str:
    """Remove emojis from text."""
    return remove_pattern(text, PATTERNS['emojis'])


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return remove_pattern(text, PATTERNS['punctuation'])


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(PATTERNS['extra_whitespace'], ' ', text).strip()


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_short_words(text: str, min_length: int = 2) -> str:
    """
    Remove words shorter than minimum length.
    
    Args:
        text: Input text
        min_length: Minimum word length
        
    Returns:
        Text with short words removed
    """
    words = text.split()
    filtered_words = [word for word in words if len(word) >= min_length]
    return ' '.join(filtered_words)


def remove_stopwords(text: str, stopwords: Optional[List[str]] = None) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Input text
        stopwords: List of stopwords (uses NLTK default if None)
        
    Returns:
        Text with stopwords removed
    """
    if stopwords is None:
        try:
            import nltk
            from nltk.corpus import stopwords as nltk_stopwords
            nltk.download('stopwords', quiet=True)
            stopwords = set(nltk_stopwords.words('english'))
        except ImportError:
            logging.warning("NLTK not available, using basic stopwords")
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)


# Create cleaning pipeline functions
basic_clean = compose(
    normalize_whitespace,
    to_lowercase,
    remove_urls,
    remove_mentions,
    remove_emails
)

aggressive_clean = compose(
    normalize_whitespace,
    partial(remove_short_words, min_length=3),
    remove_stopwords,
    remove_punctuation,
    remove_numbers,
    remove_emojis,
    remove_hashtags,
    remove_urls,
    remove_mentions,
    remove_emails,
    to_lowercase
)

moderate_clean = compose(
    normalize_whitespace,
    partial(remove_short_words, min_length=2),
    remove_urls,
    remove_mentions,
    remove_emails,
    to_lowercase
)


def create_custom_cleaner(*cleaning_functions: Callable[[str], str]) -> Callable[[str], str]:
    """
    Create a custom text cleaner by composing cleaning functions.
    
    Args:
        *cleaning_functions: Functions to compose into cleaner
        
    Returns:
        Composed cleaning function
    """
    return compose(*cleaning_functions)


def clean_text_series(series: pd.Series, cleaner: Callable[[str], str] = moderate_clean) -> pd.Series:
    """
    Clean a pandas Series of text using specified cleaner.
    
    Args:
        series: Pandas Series containing text
        cleaner: Cleaning function to apply
        
    Returns:
        Cleaned Series
    """
    return series.astype(str).apply(cleaner)


def clean_tweets_dataframe(df: pd.DataFrame, text_column: str = 'text', 
                          cleaner: Callable[[str], str] = moderate_clean,
                          create_new_column: bool = True) -> pd.DataFrame:
    """
    Clean tweets in a DataFrame.
    
    Args:
        df: DataFrame containing tweets
        text_column: Name of column containing text
        cleaner: Cleaning function to apply
        create_new_column: Whether to create new column or overwrite existing
        
    Returns:
        DataFrame with cleaned text
    """
    if df.empty or text_column not in df.columns:
        logging.warning(f"Column '{text_column}' not found or DataFrame is empty")
        return df
    
    df_cleaned = df.copy()
    
    cleaned_text = clean_text_series(df_cleaned[text_column], cleaner)
    
    if create_new_column:
        df_cleaned[f'{text_column}_cleaned'] = cleaned_text
    else:
        df_cleaned[text_column] = cleaned_text
    
    if create_new_column:
        df_cleaned = df_cleaned[df_cleaned[f'{text_column}_cleaned'].str.strip() != '']
    else:
        df_cleaned = df_cleaned[df_cleaned[text_column].str.strip() != '']
    
    logging.info(f"Cleaned {len(df_cleaned)} tweets (removed {len(df) - len(df_cleaned)} empty texts)")
    return df_cleaned


def validate_cleaned_text(text: str, min_length: int = 10) -> bool:
    """
    Validate that cleaned text meets minimum requirements.
    
    Args:
        text: Cleaned text to validate
        min_length: Minimum text length
        
    Returns:
        True if text is valid, False otherwise
    """
    if not isinstance(text, str):
        return False
    
    text = text.strip()
    if len(text) < min_length:
        return False
    
    word_count = len(text.split())
    return word_count >= 2


def filter_valid_tweets(df: pd.DataFrame, text_column: str = 'text_cleaned', 
                       min_length: int = 10) -> pd.DataFrame:
    """
    Filter DataFrame to keep only tweets with valid cleaned text.
    
    Args:
        df: DataFrame to filter
        text_column: Column containing cleaned text
        min_length: Minimum text length
        
    Returns:
        Filtered DataFrame
    """
    if df.empty or text_column not in df.columns:
        return df
    
    valid_mask = df[text_column].apply(lambda x: validate_cleaned_text(x, min_length))
    df_filtered = df[valid_mask].copy()
    
    logging.info(f"Filtered to {len(df_filtered)} valid tweets (removed {len(df) - len(df_filtered)} invalid)")
    return df_filtered

sentiment_cleaner = create_custom_cleaner(
    normalize_whitespace,
    remove_urls,
    remove_mentions,
    remove_emails,
    to_lowercase
)

topic_modeling_cleaner = create_custom_cleaner(
    normalize_whitespace,
    partial(remove_short_words, min_length=3),
    remove_stopwords,
    remove_punctuation,
    remove_numbers,
    remove_urls,
    remove_mentions,
    remove_emails,
    to_lowercase
)

ner_cleaner = create_custom_cleaner(
    normalize_whitespace,
    remove_urls,
    remove_emails
)
