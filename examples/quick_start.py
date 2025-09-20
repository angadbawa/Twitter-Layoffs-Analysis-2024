import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from twitter_analysis.data import scrape_layoff_tweets, save_tweets_csv
from twitter_analysis.preprocessing import clean_tweets_dataframe, moderate_clean
from twitter_analysis.analysis.sentiment import analyze_tweets_sentiment
from twitter_analysis.visualization import create_sentiment_distribution_plot
from twitter_analysis.utils.helpers import pipe, setup_logging, timer
from twitter_analysis.utils.helpers import memoize
import logging
from functools import partial
    from twitter_analysis.analysis.sentiment import filter_by_sentiment


@timer
def functional_analysis_pipeline(num_tweets: int = 50) -> None:
    """
    Demonstrate functional programming pipeline for Twitter analysis.
    
    Args:
        num_tweets: Number of tweets to analyze
    """
    print("🚀 Starting Twitter Layoffs Analysis Pipeline")
    
    # Functional pipeline using pipe utility
    result = pipe(
        num_tweets,
        # Step 1: Data collection
        lambda n: scrape_layoff_tweets(num_tweets=n, hashtags=['layoffs', 'jobcuts']),
        
        # Step 2: Data validation
        lambda df: df if not df.empty else print("❌ No data collected") or df,
        
        # Step 3: Text preprocessing
        lambda df: clean_tweets_dataframe(df, cleaner=moderate_clean) if not df.empty else df,
        
        # Step 4: Sentiment analysis
        lambda df: analyze_tweets_sentiment(df) if not df.empty else df,
        
        # Step 5: Filter high-confidence results
        lambda df: df[df['sentiment_score'] > 0.6] if 'sentiment_score' in df.columns else df
    )
    
    if result.empty:
        print("❌ Pipeline failed - no data to analyze")
        return
    
    # Display results
    print(f"\n📊 Analysis Results:")
    print(f"   • Total tweets analyzed: {len(result)}")
    print(f"   • Sentiment distribution:")
    
    if 'sentiment_label' in result.columns:
        sentiment_counts = result['sentiment_label'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(result)) * 100
            print(f"     - {sentiment}: {count} ({percentage:.1f}%)")
    
    # Save results
    save_tweets_csv(result, "analysis_results.csv")
    print(f"\n💾 Results saved to: analysis_results.csv")
    
    # Create visualization
    try:
        create_sentiment_distribution_plot(result)
        print(f"📈 Visualization saved to: output/sentiment_distribution.png")
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
    
    print("\n✅ Pipeline completed successfully!")


def demonstrate_functional_features():
    """Demonstrate functional programming features."""
    
    print("\n🔧 Functional Programming Features Demo:")
    
    # 1. Function composition
    from twitter_analysis.preprocessing.cleaner import compose, remove_urls, remove_mentions, to_lowercase
    
    # Create a custom cleaner by composing functions
    custom_cleaner = compose(to_lowercase, remove_mentions, remove_urls)
    
    sample_text = "Check out this link https://example.com @user #hashtag"
    cleaned = custom_cleaner(sample_text)
    print(f"   • Function composition:")
    print(f"     Original: {sample_text}")
    print(f"     Cleaned:  {cleaned}")

    get_positive_tweets = partial(filter_by_sentiment, sentiment='POSITIVE', min_confidence=0.8)
    print(f"   • Partial application: Created specialized filter function")
    
    @memoize
    def expensive_computation(x):
        return x ** 2
    
    print(f"   • Memoization: Cached function results for performance")


def main():
    """Main function to run the quick start example."""
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    print("=" * 60)
    print("🐦 Twitter Layoffs Analysis - Quick Start Example")
    print("=" * 60)
    
    try:
        # Run functional analysis pipeline
        functional_analysis_pipeline(num_tweets=20)  # Small sample for demo
        
        # Demonstrate functional features
        demonstrate_functional_features()
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        logging.error(f"Quick start failed: {e}")
    
    print("\n" + "=" * 60)
    print("📚 For more examples, check the notebooks/ directory")
    print("🐳 To run with Docker: docker-compose up")
    print("📖 Full documentation: README.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
