"""
Script for performing sentiment analysis on GDELT data.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.nlp_utils import SentimentAnalyzer
from utils.plot_utils import plot_sentiment_index, save_figure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """Load GDELT macro events data with trading day alignment."""
    logger.info("Loading GDELT macro events with trading day mapping...")
    df = pd.read_csv(
        "data/raw/gdelt/gdelt_macro_events_with_headlines.csv",
        parse_dates=["ts_gdelt_utc", "ts_article_utc", "ts_fetch_utc", "added_dt_utc", "trading_day_fx", "trading_day_zn"]
    )
    logger.info(f"Loaded {len(df):,} events with trading day mapping")
    return df

def init_finbert():
    """Initialize FinBERT model and tokenizer."""
    logger.info("Initializing FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model

def score_batch(texts, tokenizer, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Score a batch of texts using FinBERT."""
    if not texts:  # Handle empty batch
        return []
    
    # Tokenize and prepare inputs
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
    
    # Calculate sentiment score (positive - negative)
    scores = probs[:, 0] - probs[:, 2]  # positive - negative
    return scores.cpu().numpy()

def process_sentiment(df, tokenizer, model, batch_size=64):
    """Process sentiment for all events in batches with fallback to avg_tone."""
    logger.info("Processing sentiment scores with fallback strategy...")
    df['text'] = df['headline'].astype(str)

    # Only process rows with meaningful headlines (not empty or "nan")
    has_headline = (df['text'].str.strip().ne("")) & (df['text'] != "nan")
    finbert_scores = np.full(len(df), np.nan, dtype=float)
    
    headline_count = has_headline.sum()
    fallback_count = (~has_headline).sum()
    logger.info(f"Events with headlines: {headline_count:,} ({headline_count/len(df):.1%})")
    logger.info(f"Events using fallback: {fallback_count:,} ({fallback_count/len(df):.1%})")

    # Only run FinBERT on rows with headlines (saves computation)
    idx = np.where(has_headline.values)[0]
    for i in tqdm(range(0, len(idx), batch_size)):
        sl = idx[i:i+batch_size]
        texts = df['text'].iloc[sl].tolist()
        finbert_scores[sl] = score_batch(texts, tokenizer, model)  # Use default device

    df['finbert_score'] = finbert_scores
    
    # Apply fallback: replace NaN/empty headline scores with avg_tone (scaled to [-1,1])
    fallback_mask = ~has_headline
    if fallback_mask.any():
        # Find available tone column
        tone_col = "avg_tone" if "avg_tone" in df.columns else ("tone" if "tone" in df.columns else None)
        if tone_col is None:
            raise ValueError("No avg_tone/tone column for fallback.")
        # Core fix: /5 instead of /100 for better scale
        fallback_scores = np.tanh(df.loc[fallback_mask, tone_col].astype(float) / 5.0)
        df.loc[fallback_mask, 'finbert_score'] = fallback_scores
        logger.info(f"Applied {tone_col} fallback to {fallback_mask.sum():,} events")
    
    return df

def aggregate_daily(df):
    """Aggregate sentiment scores to trading day frequency using FX trading day mapping."""
    logger.info("Aggregating to trading day frequency using trading_day_fx...")
    
    # Handle different Goldstein column names for robustness
    gold_col = "goldstein" if "goldstein" in df.columns else (
        "goldstein_scale" if "goldstein_scale" in df.columns else None
    )
    if gold_col:
        logger.info(f"Using Goldstein column: {gold_col}")
    else:
        logger.warning("No Goldstein column found - skipping Goldstein aggregation")
    
    # Use trading_day_fx for aggregation (22:00 UTC cutoff)
    # Convert trading_day_fx to date for grouping
    df['trading_date'] = df['trading_day_fx'].dt.date
    
    # Build aggregation map dynamically
    agg_map = {
        'finbert_score': ['mean', 'std', 'size']
    }
    if gold_col:
        agg_map[gold_col] = ['mean', 'std']
    
    daily = df.groupby('trading_date').agg(agg_map).reset_index()
    
    # Restore column names
    daily.columns = ['date', 'sentiment_mean', 'sentiment_std', 'article_count'] + (
        ['goldstein_mean', 'goldstein_std'] if gold_col else []
    )
    
    # Sort by date and add rolling averages
    daily = daily.sort_values('date')
    daily['sentiment_ma5'] = daily['sentiment_mean'].rolling(5, min_periods=1).mean()
    daily['sentiment_ma20'] = daily['sentiment_mean'].rolling(20, min_periods=1).mean()
    
    logger.info(f"Aggregated to {len(daily)} trading days")
    logger.info(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    
    return daily

def save_results(daily_df):
    """Save processed results."""
    # Create directory if it doesn't exist
    os.makedirs("data/processed/sentiment", exist_ok=True)
    
    # Save daily sentiment
    output_path = "data/processed/sentiment/news_sentiment_daily.csv"
    daily_df.to_csv(output_path, index=False)
    logger.info(f"Saved trading day sentiment to {output_path}")
    
    # Save summary statistics
    summary = {
        'total_trading_days': len(daily_df),
        'avg_articles_per_day': daily_df['article_count'].mean(),
        'avg_sentiment': daily_df['sentiment_mean'].mean(),
        'sentiment_std': daily_df['sentiment_mean'].std(),
        'date_range': f"{daily_df['date'].min()} to {daily_df['date'].max()}",
        'alignment': 'trading_day_fx (22:00 UTC cutoff)'
    }
    
    logger.info("\nSummary Statistics:")
    for key, value in summary.items():
        logger.info(f"{key}: {value}")

def main():
    try:
        # 1. Load data
        df = load_data()
        
        # 2. Initialize FinBERT
        tokenizer, model = init_finbert()
        
        # 3. Process sentiment
        df = process_sentiment(df, tokenizer, model, batch_size=64)
        
        # 4. Optional deduplication to avoid duplicate URL scoring
        if "source_url" in df.columns:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=["trading_day_fx", "source_url"])
            after_dedup = len(df)
            if before_dedup != after_dedup:
                logger.info(f"Deduplicated {before_dedup - after_dedup} duplicate URL entries")
        
        # 5. Aggregate to daily frequency
        daily_df = aggregate_daily(df)
        
        # 6. Save results
        save_results(daily_df)
        
        logger.info("Sentiment analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 