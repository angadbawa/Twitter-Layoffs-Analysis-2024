from typing import Optional, Dict, Any, List
import pandas as pd
import logging
from pathlib import Path
from ..utils.helpers import safe_execute
from ..utils.config import get_output_path, get_config


@safe_execute
def load_tweets_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load tweets from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with tweets or None if failed
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} tweets from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load tweets from {file_path}: {e}")
        return None


@safe_execute
def save_tweets_csv(df: pd.DataFrame, filename: str = None) -> bool:
    """
    Save tweets DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filename: Optional filename (uses config default if None)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if filename is None:
            filename = get_config("data.tweets_file")
        
        file_path = get_output_path(filename)
        df.to_csv(file_path, index=False)
        logging.info(f"Saved {len(df)} tweets to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save tweets: {e}")
        return False


def load_or_create_tweets(file_path: str = None, scrape_if_missing: bool = True) -> pd.DataFrame:
    """
    Load tweets from file or create new dataset if file doesn't exist.
    
    Args:
        file_path: Path to tweets file
        scrape_if_missing: Whether to scrape new data if file is missing
        
    Returns:
        DataFrame with tweets
    """
    if file_path is None:
        file_path = get_output_path(get_config("data.tweets_file"))
    
    # Try to load existing file
    if Path(file_path).exists():
        df = load_tweets_csv(file_path)
        if df is not None and not df.empty:
            return df
    
    # Create new dataset if file doesn't exist or is empty
    if scrape_if_missing:
        logging.info("No existing tweets file found. Scraping new data...")
        from .scraper import scrape_layoff_tweets
        df = scrape_layoff_tweets()
        
        if not df.empty:
            save_tweets_csv(df, Path(file_path).name)
        
        return df
    else:
        logging.warning("No tweets file found and scraping is disabled")
        return pd.DataFrame()


def validate_tweets_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns for tweet analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['text', 'date', 'user']
    
    if df.empty:
        logging.error("DataFrame is empty")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for empty text column
    if df['text'].isna().all() or (df['text'].str.strip() == '').all():
        logging.error("All text entries are empty")
        return False
    
    logging.info("DataFrame validation passed")
    return True


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    if df.empty:
        return {"error": "DataFrame is empty"}
    
    info = {
        "total_tweets": len(df),
        "unique_users": df['user'].nunique() if 'user' in df.columns else 0,
        "date_range": {
            "start": df['date'].min() if 'date' in df.columns else None,
            "end": df['date'].max() if 'date' in df.columns else None
        },
        "columns": list(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "null_counts": df.isnull().sum().to_dict()
    }
    
    # Add hashtag distribution if available
    if 'source_hashtag' in df.columns:
        info["hashtag_distribution"] = df['source_hashtag'].value_counts().to_dict()
    
    return info


def sample_tweets(df: pd.DataFrame, n: int = 100, random_state: int = 42) -> pd.DataFrame:
    """
    Get a random sample of tweets.
    
    Args:
        df: DataFrame to sample from
        n: Number of samples
        random_state: Random state for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    if df.empty:
        return df
    
    sample_size = min(n, len(df))
    return df.sample(n=sample_size, random_state=random_state)


def filter_tweets_by_date(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Filter tweets by date range.
    
    Args:
        df: DataFrame to filter
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        
    Returns:
        Filtered DataFrame
    """
    if df.empty or 'date' not in df.columns:
        return df
    
    # Convert date column to datetime
    df_filtered = df.copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
    
    # Apply date filters
    if start_date:
        start_date = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered['date'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered['date'] <= end_date]
    
    logging.info(f"Filtered to {len(df_filtered)} tweets within date range")
    return df_filtered


def export_tweets_json(df: pd.DataFrame, filename: str = "tweets.json") -> bool:
    """
    Export tweets to JSON format.
    
    Args:
        df: DataFrame to export
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = get_output_path(filename)
        df.to_json(file_path, orient='records', indent=2)
        logging.info(f"Exported {len(df)} tweets to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to export tweets to JSON: {e}")
        return False
