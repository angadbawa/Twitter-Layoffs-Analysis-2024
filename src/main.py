import argparse
import logging
from pathlib import Path
from typing import Optional, List

from twitter_analysis.utils.helpers import setup_logging, pipe
from twitter_analysis.utils.config import get_config, update_config
from twitter_analysis.data import scrape_layoff_tweets, load_or_create_tweets, save_tweets_csv
from twitter_analysis.preprocessing import clean_tweets_dataframe, moderate_clean
from twitter_analysis.analysis.sentiment import analyze_tweets_sentiment
from twitter_analysis.visualization import plots


def run_data_collection(num_tweets: int = None, hashtags: List[str] = None) -> None:
    """
    Run data collection pipeline.
    
    Args:
        num_tweets: Number of tweets to collect
        hashtags: List of hashtags to search
    """
    logging.info("Starting data collection...")
    
    # Scrape tweets
    df = scrape_layoff_tweets(num_tweets=num_tweets, hashtags=hashtags)
    
    if df.empty:
        logging.error("No tweets collected")
        return
    
    # Save raw data
    save_tweets_csv(df, "raw_tweets.csv")
    logging.info(f"Collected and saved {len(df)} tweets")


def run_preprocessing_pipeline(input_file: str = None, output_file: str = None) -> None:
    """
    Run preprocessing pipeline.
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
    """
    logging.info("Starting preprocessing pipeline...")
    
    # Load data
    df = load_or_create_tweets(input_file, scrape_if_missing=False)
    
    if df.empty:
        logging.error("No data to preprocess")
        return
    
    # Clean tweets
    df_cleaned = pipe(
        df,
        lambda x: clean_tweets_dataframe(x, cleaner=moderate_clean),
        lambda x: x.dropna(subset=['text_cleaned']),
        lambda x: x[x['text_cleaned'].str.len() > 10]  # Filter very short texts
    )
    
    # Save cleaned data
    output_file = output_file or "cleaned_tweets.csv"
    save_tweets_csv(df_cleaned, output_file)
    logging.info(f"Preprocessed and saved {len(df_cleaned)} tweets")


def run_analysis_pipeline(input_file: str = None) -> None:
    """
    Run complete analysis pipeline.
    
    Args:
        input_file: Input CSV file path
    """
    logging.info("Starting analysis pipeline...")

    df = load_or_create_tweets(input_file, scrape_if_missing=False)
    
    if df.empty:
        logging.error("No data to analyze")
        return

    if 'text_cleaned' not in df.columns:
        df = clean_tweets_dataframe(df, cleaner=moderate_clean)

    df_analyzed = analyze_tweets_sentiment(df)
    save_tweets_csv(df_analyzed, "analyzed_tweets.csv")
    
    try:
        plots.create_sentiment_distribution_plot(df_analyzed)
        plots.create_sentiment_timeline_plot(df_analyzed)
        logging.info("Generated visualizations")
    except Exception as e:
        logging.warning(f"Failed to generate some visualizations: {e}")
    
    logging.info(f"Completed analysis for {len(df_analyzed)} tweets")


def run_full_pipeline(num_tweets: int = None, hashtags: List[str] = None) -> None:
    """
    Run complete end-to-end pipeline.
    
    Args:
        num_tweets: Number of tweets to collect
        hashtags: List of hashtags to search
    """
    logging.info("Starting full pipeline...")
    
    run_data_collection(num_tweets, hashtags)
    run_preprocessing_pipeline("raw_tweets.csv", "tweets.csv")
    run_analysis_pipeline("tweets.csv")
    
    logging.info("Full pipeline completed successfully!")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Twitter Layoffs Analysis")
    
    parser.add_argument(
        "--mode",
        choices=["collect", "preprocess", "analyze", "full"],
        default="full",
        help="Pipeline mode to run"
    )
    
    parser.add_argument(
        "--num-tweets",
        type=int,
        default=None,
        help="Number of tweets to collect"
    )
    
    parser.add_argument(
        "--hashtags",
        nargs="+",
        default=None,
        help="Hashtags to search for"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input CSV file path"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path"
    )
    
    args = parser.parse_args()
    log_level = getattr(logging, args.log_level)
    setup_logging(args.log_file, log_level)
    
    logging.info("Starting Twitter Layoffs Analysis")
    logging.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == "collect":
            run_data_collection(args.num_tweets, args.hashtags)
        elif args.mode == "preprocess":
            run_preprocessing_pipeline(args.input_file, args.output_file)
        elif args.mode == "analyze":
            run_analysis_pipeline(args.input_file)
        elif args.mode == "full":
            run_full_pipeline(args.num_tweets, args.hashtags)
        
        logging.info("Application completed successfully")
        
    except Exception as e:
        logging.error(f"Application failed: {e}")
        raise


if __name__ == "__main__":
    main()
