from typing import List, Dict, Any, Optional, Callable
from functools import partial
import pandas as pd
import logging
from ntscraper import Nitter
import tweepy
from ..utils.helpers import timer, safe_execute, memoize
from ..utils.config import get_config

def initialize_nitter_scraper(log_level: int = 1, skip_instance_check: bool = False) -> Nitter:
    """
    Initialize Nitter scraper with configuration.
    
    Args:
        log_level: Logging level for scraper
        skip_instance_check: Whether to skip instance check
        
    Returns:
        Configured Nitter scraper instance
    """
    return Nitter(log_level=log_level, skip_instance_check=skip_instance_check)


@timer
def scrape_tweets_by_hashtag(scraper: Nitter, hashtag: str, num_tweets: int) -> List[Dict[str, Any]]:
    """
    Scrape tweets by hashtag using functional approach.
    
    Args:
        scraper: Nitter scraper instance
        hashtag: Hashtag to search for
        num_tweets: Number of tweets to scrape
        
    Returns:
        List of tweet dictionaries
    """
    try:
        tweets = scraper.get_tweets(hashtag, mode="hashtag", number=num_tweets)
        return tweets if tweets else []
    except Exception as e:
        logging.error(f"Failed to scrape tweets for hashtag '{hashtag}': {e}")
        return []


@timer
def scrape_tweets_by_keyword(scraper: Nitter, keyword: str, num_tweets: int) -> List[Dict[str, Any]]:
    """
    Scrape tweets by keyword search.
    
    Args:
        scraper: Nitter scraper instance
        keyword: Keyword to search for
        num_tweets: Number of tweets to scrape
        
    Returns:
        List of tweet dictionaries
    """
    try:
        tweets = scraper.get_tweets(keyword, mode="term", number=num_tweets)
        return tweets if tweets else []
    except Exception as e:
        logging.error(f"Failed to scrape tweets for keyword '{keyword}': {e}")
        return []


def create_tweet_scraper(scraper: Nitter, mode: str = "hashtag") -> Callable[[str, int], List[Dict[str, Any]]]:
    """
    Create a specialized tweet scraper function.
    
    Args:
        scraper: Nitter scraper instance
        mode: Scraping mode ('hashtag' or 'keyword')
        
    Returns:
        Specialized scraper function
    """
    if mode == "hashtag":
        return partial(scrape_tweets_by_hashtag, scraper)
    elif mode == "keyword":
        return partial(scrape_tweets_by_keyword, scraper)
    else:
        raise ValueError(f"Unsupported scraping mode: {mode}")


@timer
def scrape_multiple_hashtags(scraper: Nitter, hashtags: List[str], tweets_per_hashtag: int) -> List[Dict[str, Any]]:
    """
    Scrape tweets from multiple hashtags.
    
    Args:
        scraper: Nitter scraper instance
        hashtags: List of hashtags to scrape
        tweets_per_hashtag: Number of tweets per hashtag
        
    Returns:
        Combined list of all tweets
    """
    hashtag_scraper = create_tweet_scraper(scraper, "hashtag")
    
    all_tweets = []
    for hashtag in hashtags:
        logging.info(f"Scraping tweets for hashtag: {hashtag}")
        tweets = hashtag_scraper(hashtag, tweets_per_hashtag)
        
        # Add hashtag metadata to tweets
        for tweet in tweets:
            tweet['source_hashtag'] = hashtag
        
        all_tweets.extend(tweets)
        logging.info(f"Scraped {len(tweets)} tweets for hashtag: {hashtag}")
    
    return all_tweets


def extract_tweet_features(tweet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant features from a tweet dictionary.
    
    Args:
        tweet: Raw tweet dictionary
        
    Returns:
        Dictionary with extracted features
    """
    return {
        'text': tweet.get('text', ''),
        'date': tweet.get('date', ''),
        'user': tweet.get('user', {}).get('name', ''),
        'username': tweet.get('user', {}).get('username', ''),
        'likes': tweet.get('stats', {}).get('likes', 0),
        'retweets': tweet.get('stats', {}).get('retweets', 0),
        'replies': tweet.get('stats', {}).get('replies', 0),
        'source_hashtag': tweet.get('source_hashtag', ''),
        'url': tweet.get('url', '')
    }


@timer
def process_scraped_tweets(tweets: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process scraped tweets into a clean DataFrame.
    
    Args:
        tweets: List of raw tweet dictionaries
        
    Returns:
        Processed DataFrame
    """
    if not tweets:
        logging.warning("No tweets to process")
        return pd.DataFrame()
    
    # Extract features from all tweets
    processed_tweets = [extract_tweet_features(tweet) for tweet in tweets]
    
    # Create DataFrame
    df = pd.DataFrame(processed_tweets)
    
    # Remove duplicates based on text
    df = df.drop_duplicates(subset=['text'], keep='first')
    
    # Remove empty texts
    df = df[df['text'].str.strip() != '']
    
    logging.info(f"Processed {len(df)} unique tweets")
    return df


@timer
def scrape_layoff_tweets(num_tweets: int = None, hashtags: List[str] = None) -> pd.DataFrame:
    """
    Main function to scrape layoff-related tweets.
    
    Args:
        num_tweets: Total number of tweets to scrape (distributed across hashtags)
        hashtags: List of hashtags to scrape
        
    Returns:
        DataFrame with scraped tweets
    """
    # Get configuration
    config = get_config("data")
    num_tweets = num_tweets or config["max_tweets"]
    hashtags = hashtags or config["hashtags"]
    
    # Initialize scraper
    scraper = initialize_nitter_scraper()
    if not scraper:
        logging.error("Failed to initialize scraper")
        return pd.DataFrame()
    
    # Calculate tweets per hashtag
    tweets_per_hashtag = num_tweets // len(hashtags)
    
    # Scrape tweets
    logging.info(f"Starting to scrape {num_tweets} tweets from {len(hashtags)} hashtags")
    raw_tweets = scrape_multiple_hashtags(scraper, hashtags, tweets_per_hashtag)
    
    # Process tweets
    df = process_scraped_tweets(raw_tweets)
    
    logging.info(f"Successfully scraped and processed {len(df)} tweets")
    return df

scrape_hashtag_tweets = partial(scrape_layoff_tweets, hashtags=["layoffs"])
scrape_jobcut_tweets = partial(scrape_layoff_tweets, hashtags=["jobcuts"])
scrape_recession_tweets = partial(scrape_layoff_tweets, hashtags=["recession"])
