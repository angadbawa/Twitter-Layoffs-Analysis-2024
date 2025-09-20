from typing import Dict, Any, Optional
from functools import partial
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "data": {
        "output_dir": "output",
        "tweets_file": "tweets.csv",
        "max_tweets": 1000,
        "hashtags": ["layoffs", "jobcuts", "unemployment", "recession"]
    },
    "models": {
        "sentiment_model": "nlptown/bert-base-multilingual-uncased-sentiment",
        "emotion_model": "distilbert-base-uncased-finetuned-sst-2-english",
        "spacy_model": "en_core_web_sm"
    },
    "analysis": {
        "num_topics": 5,
        "max_features": 1000,
        "min_df": 2,
        "max_df": 0.8,
        "random_state": 42
    },
    "visualization": {
        "figure_size": (12, 8),
        "color_palette": "viridis",
        "dpi": 300
    }
}

def get_config(key_path: Optional[str] = None) -> Any:
    """
    Get configuration value by key path.
    
    Args:
        key_path: Dot-separated path to config value (e.g., 'data.max_tweets')
        
    Returns:
        Configuration value or entire config if key_path is None
    """
    config = DEFAULT_CONFIG.copy()
    
    if key_path is None:
        return config
    
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(f"Configuration key '{key_path}' not found")
    
    return value


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of updates to apply
        
    Returns:
        Updated configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    return deep_update(config, updates)


def get_output_path(filename: str) -> Path:
    """
    Get full output path for a file.
    
    Args:
        filename: Name of the file
        
    Returns:
        Full path to output file
    """
    output_dir = Path(get_config("data.output_dir"))
    output_dir.mkdir(exist_ok=True)
    return output_dir / filename

get_data_config = partial(get_config, "data")
get_model_config = partial(get_config, "models")
get_analysis_config = partial(get_config, "analysis")
get_viz_config = partial(get_config, "visualization")
