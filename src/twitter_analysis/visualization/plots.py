from typing import Optional, Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path

from ..utils.config import get_config, get_output_path


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_sentiment_distribution_plot(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create sentiment distribution visualization.
    
    Args:
        df: DataFrame with sentiment analysis results
        save_path: Optional path to save the plot
    """
    if df.empty or 'sentiment_label' not in df.columns:
        logging.warning("No sentiment data available for plotting")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    sentiment_counts = df['sentiment_label'].value_counts()
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    ax1.set_title('Sentiment Distribution')
    
    # Bar plot with confidence scores
    if 'sentiment_score' in df.columns:
        sns.boxplot(data=df, x='sentiment_label', y='sentiment_score', ax=ax2)
        ax2.set_title('Sentiment Confidence Scores')
        ax2.set_ylabel('Confidence Score')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = get_output_path("sentiment_distribution.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved sentiment distribution plot to {save_path}")


def create_sentiment_timeline_plot(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create sentiment timeline visualization.
    
    Args:
        df: DataFrame with sentiment and date columns
        save_path: Optional path to save the plot
    """
    if df.empty or 'sentiment_label' not in df.columns or 'date' not in df.columns:
        logging.warning("No sentiment timeline data available for plotting")
        return
    
    # Prepare data
    df_plot = df.copy()
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    
    # Group by date and sentiment
    timeline_data = df_plot.groupby([
        df_plot['date'].dt.date,
        'sentiment_label'
    ]).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    timeline_data.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    ax.set_title('Sentiment Timeline')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Tweets')
    ax.legend(title='Sentiment')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path is None:
        save_path = get_output_path("sentiment_timeline.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved sentiment timeline plot to {save_path}")


def create_interactive_sentiment_plot(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create interactive sentiment visualization using Plotly.
    
    Args:
        df: DataFrame with sentiment analysis results
        save_path: Optional path to save the plot
    """
    if df.empty or 'sentiment_label' not in df.columns:
        logging.warning("No sentiment data available for plotting")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sentiment Distribution', 'Confidence Scores', 
                       'Sentiment Timeline', 'Top Words by Sentiment'),
        specs=[[{"type": "pie"}, {"type": "box"}],
               [{"type": "scatter", "colspan": 2}, None]]
    )
    
    # Pie chart
    sentiment_counts = df['sentiment_label'].value_counts()
    fig.add_trace(
        go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
               name="Sentiment Distribution"),
        row=1, col=1
    )
    
    # Box plot for confidence scores
    if 'sentiment_score' in df.columns:
        for sentiment in df['sentiment_label'].unique():
            sentiment_data = df[df['sentiment_label'] == sentiment]
            fig.add_trace(
                go.Box(y=sentiment_data['sentiment_score'], name=sentiment),
                row=1, col=2
            )
    
    # Timeline
    if 'date' in df.columns:
        df_timeline = df.copy()
        df_timeline['date'] = pd.to_datetime(df_timeline['date'])
        
        timeline_data = df_timeline.groupby([
            df_timeline['date'].dt.date,
            'sentiment_label'
        ]).size().reset_index(name='count')
        
        for sentiment in timeline_data['sentiment_label'].unique():
            sentiment_timeline = timeline_data[timeline_data['sentiment_label'] == sentiment]
            fig.add_trace(
                go.Scatter(
                    x=sentiment_timeline['date'],
                    y=sentiment_timeline['count'],
                    mode='lines+markers',
                    name=f'{sentiment} Timeline'
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title_text="Twitter Sentiment Analysis Dashboard",
        showlegend=True,
        height=800
    )
    
    # Save plot
    if save_path is None:
        save_path = get_output_path("interactive_sentiment_dashboard.html")
    
    fig.write_html(save_path)
    logging.info(f"Saved interactive sentiment dashboard to {save_path}")


def create_wordcloud_plot(df: pd.DataFrame, sentiment: str = None, save_path: str = None) -> None:
    """
    Create word cloud visualization.
    
    Args:
        df: DataFrame with text data
        sentiment: Optional sentiment to filter by
        save_path: Optional path to save the plot
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logging.warning("WordCloud not available, skipping word cloud generation")
        return
    
    text_column = 'text_cleaned' if 'text_cleaned' in df.columns else 'text'
    
    if df.empty or text_column not in df.columns:
        logging.warning("No text data available for word cloud")
        return
    
    # Filter by sentiment if specified
    df_filtered = df.copy()
    if sentiment and 'sentiment_label' in df.columns:
        df_filtered = df_filtered[df_filtered['sentiment_label'] == sentiment.upper()]
    
    # Combine all text
    text = ' '.join(df_filtered[text_column].astype(str))
    
    if not text.strip():
        logging.warning("No text available after filtering")
        return
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(text)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    title = f'Word Cloud - {sentiment} Sentiment' if sentiment else 'Word Cloud - All Tweets'
    plt.title(title)
    
    # Save plot
    if save_path is None:
        filename = f"wordcloud_{sentiment.lower()}.png" if sentiment else "wordcloud_all.png"
        save_path = get_output_path(filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved word cloud to {save_path}")


def create_comprehensive_dashboard(df: pd.DataFrame) -> None:
    """
    Create a comprehensive analysis dashboard.
    
    Args:
        df: DataFrame with complete analysis results
    """
    logging.info("Creating comprehensive dashboard...")
    
    # Create individual plots
    create_sentiment_distribution_plot(df)
    create_sentiment_timeline_plot(df)
    create_interactive_sentiment_plot(df)
    
    # Create word clouds for each sentiment
    if 'sentiment_label' in df.columns:
        for sentiment in df['sentiment_label'].unique():
            create_wordcloud_plot(df, sentiment)
    
    # Create overall word cloud
    create_wordcloud_plot(df)
    
    logging.info("Comprehensive dashboard created successfully")


def save_plot_config() -> Dict[str, Any]:
    """Get plotting configuration."""
    return get_config("visualization")
