#!/usr/bin/env python
# scripts/fetch_gdelt_data.py

import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from gdelt import gdelt
import os
import warnings
from bs4 import GuessedAtParserWarning, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.headline_utils import fetch_headline
from concurrent.futures import ThreadPoolExecutor, as_completed

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def fetch_all_events(start: str, end: str) -> pd.DataFrame:
    client = gdelt(version=2)
    logger.info(f"Requesting GDELT events from {start} to {end}...")
    df = client.Search([start, end], table="events", normcols=True)
    logger.info(f"Fetched {len(df):,} total events.")
    return df

def filter_and_process(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Unique eventcode values before filtering: {df.eventcode.unique()}")
    macro_code_prefixes = tuple(str(i).zfill(3) for i in range(100, 200))
    macro_df = df[df.eventcode.str.startswith(macro_code_prefixes, na=False)]
    logger.info(f"Events remaining after macro filter: {len(macro_df):,}")

    # LONG-TERM FIX: Preserve timestamp information for proper trading day alignment
    macro_df = (
        macro_df.rename(columns={
            "sqldate": "date",
            "actor1name": "actor1",
            "actor2name": "actor2",
            "eventcode": "event_code",
            "goldsteinscale": "goldstein",
            "nummentions": "n_mentions",
            "numsources": "n_sources",
            "numarticles": "n_articles",
            "avgtone": "avg_tone",
            "sourceurl": "source_url",
            "dateadded": "timestamp_utc",  # NEW: Preserve UTC timestamp
        })
        .loc[:, ["date","actor1","actor2","event_code","goldstein",
                 "n_mentions","n_sources","n_articles","avg_tone","source_url","timestamp_utc"]]
    )
    
    # Convert sqldate to event calendar date (for backward compatibility)
    macro_df["event_date"] = pd.to_datetime(macro_df["date"], format="%Y%m%d", utc=True, errors="coerce").dt.date
    
    # Convert dateadded to proper tz-aware UTC timestamp
    # Format: YYYYMMDDHHMMSS (e.g., 20240101234500)
    macro_df["added_dt_utc"] = pd.to_datetime(macro_df["timestamp_utc"], 
                                               format="%Y%m%d%H%M%S", 
                                               utc=True, errors="coerce")
    
    # Measure missing timestamps and drop them (no imputation to avoid bias)
    missing_ratio = macro_df["added_dt_utc"].isna().mean()
    logger.info(f"Missing DATEADDED ratio: {missing_ratio:.3%}")
    macro_df = macro_df.dropna(subset=["added_dt_utc"])
    
    # Add critical timestamp fields for trading day alignment
    macro_df["ts_gdelt_utc"] = macro_df["added_dt_utc"]  # GDELT's 15-min batch time
    macro_df["ts_article_utc"] = macro_df["added_dt_utc"]  # Will be updated with meta published_time if available
    macro_df["ts_fetch_utc"] = pd.Timestamp.now(tz="UTC")  # Current fetch time for debugging
    
    # Keep only essential columns with clear naming
    keep_cols = ["ts_gdelt_utc", "ts_article_utc", "ts_fetch_utc", "added_dt_utc", "event_date", 
                 "event_code", "goldstein", "n_mentions", "n_sources", "n_articles", "avg_tone",
                 "actor1", "actor2", "source_url"]
    macro_df = macro_df[keep_cols].sort_values("added_dt_utc")
    
    logger.info(f"Preserved timestamps for {len(macro_df):,} events")
    
    # Log sample timestamps for verification
    sample_timestamps = macro_df[["event_date", "added_dt_utc", "ts_gdelt_utc"]].head(3)
    logger.info(f"Sample timestamps:\n{sample_timestamps}")
    
    return macro_df

def assign_trading_day(ts_utc: pd.Timestamp, cut_hour_utc: int = 22) -> pd.Timestamp:
    """
    Assign a UTC timestamp to its corresponding trading day based on cutoff hour.
    
    Args:
        ts_utc: UTC timestamp to classify
        cut_hour_utc: UTC hour cutoff (e.g., 22 = 22:00 UTC)
    
    Returns:
        UTC midnight timestamp of the trading day this event belongs to
        
    Example:
        - Event at 2024-01-01 21:59 UTC → belongs to 2024-01-01 trading day
        - Event at 2024-01-01 22:01 UTC → belongs to 2024-01-02 trading day
    """
    # Define the boundary for this day
    day_start = ts_utc.normalize()  # UTC 00:00
    boundary = day_start + pd.Timedelta(hours=cut_hour_utc)
    
    # Events at or after cutoff belong to the "next" trading day
    if ts_utc >= boundary:
        trading_day = day_start + pd.Timedelta(days=1)
    else:
        trading_day = day_start
    
    # Handle weekends: push to next business day
    while trading_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
        trading_day = trading_day + pd.Timedelta(days=1)
    
    # Ensure timezone awareness
    if trading_day.tz is None:
        trading_day = trading_day.tz_localize("UTC")
    
    return trading_day

def scope_top_events(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Keep only the top_n events per day by num_articles.
    Note: This groups by event_date (natural day) - will be re-done by trading day later.
    """
    df = (
        df.sort_values(["event_date","n_articles"], ascending=[True, False])
          .groupby("event_date")
          .head(top_n)
          .reset_index(drop=True)
    )
    logger.info(f"Scoped to top {top_n} events/day → {len(df):,} rows")
    return df

def scrape_headlines(urls: pd.Series, max_workers: int = 20) -> pd.Series:
    """
    Fetch headlines in parallel for a series of URLs.
    Avoids duplicate requests by deduplicating URLs first.
    Returns a Series of the same index with headlines or empty strings.
    """
    urls = urls.fillna("").astype(str)
    
    # Find unique URLs to avoid duplicate requests
    unique_urls = urls[urls != ""].drop_duplicates()
    logger.info(f"Scraping {len(unique_urls)} unique URLs from {len(urls)} total")
    
    # Fetch headlines for unique URLs only
    result = pd.Series("", index=unique_urls.index)
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(fetch_headline, url): idx for idx, url in unique_urls.items()}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result.at[idx] = fut.result()
            except Exception as e:
                logger.debug(f"Failed to fetch headline for URL {unique_urls.at[idx]}: {e}")
                result.at[idx] = ""
    
    # Map results back to original index using efficient pandas mapping
    url_to_headline = pd.Series(result.values, index=unique_urls.values).to_dict()
    out = urls.map(url_to_headline).fillna("")  # One line to handle all mapping
    
    return out

def main():
    start_date = "2015-02-18"
    end_date   = "2025-07-31"
    raw = fetch_all_events(start_date, end_date)
    proc = filter_and_process(raw)

    # 1) Don't scope to top N yet - wait until trading day aggregation
    # This avoids cutting off events that belong to the next trading day due to cutoff timing
    # We'll use filter_top_events_by_trading_day() later after proper trading day mapping
    logger.info("Skipping premature top-N filtering - will be done after trading day mapping")
    logger.info("All events preserved - top-N filtering will happen after trading day alignment")
    # proc = scope_top_events(proc, top_n=100)  # COMMENTED OUT - do this later

    # 2) Scrape headlines in monthly batches
    proc = proc.sort_values("event_date").reset_index(drop=True)
    headlines = pd.Series("", index=proc.index)

    # group by each year-month period
    periods = proc["event_date"].apply(lambda x: pd.Timestamp(x).to_period("M")).unique()
    for period in periods:
        try:
            # mask for this year-month
            mask = proc["event_date"].apply(lambda x: pd.Timestamp(x).to_period("M")) == period
            urls = proc.loc[mask, "source_url"]
            logger.info(f"Starting headline scrape for {period} ({urls.size} URLs)…")
            sub = scrape_headlines(urls, max_workers=20)  # Reduced from 50 to be nice to websites
            proc.loc[mask, "headline"] = sub
            logger.info(f"Finished headline scrape for {period}")
            
            # checkpoint this month (moved inside loop)
            ck = proc.loc[mask].copy()
            ck_path = Path(f"data/raw/gdelt/checkpoints/gdelt_headlines_{period}.csv")
            ck_path.parent.mkdir(parents=True, exist_ok=True)
            ck.to_csv(ck_path, index=False)
            logger.info(f"Saved checkpoint for {period}")
            
        except Exception as e:
            logger.error(f"Month {period} failed: {e}")
            continue

    # 3) Don't drop rows - keep all events, even without headlines
    # This prevents survivor bias - events without headlines still have avg_tone as fallback
    before = len(proc)
    missing_headlines = (proc["headline"].fillna("").str.len() == 0).sum()
    logger.info(f"Headline missing: {missing_headlines:,} / {before:,} ({missing_headlines/before:.2%})")
    logger.info("Keeping all events - missing headlines will use avg_tone as fallback feature")
    # No filtering: proc = proc[...] - keep all rows
    
    # 4) Assign trading days for proper alignment (FX uses 21:00 UTC cutoff, ZN uses 22:00 UTC cutoff)
    logger.info("Assigning trading days based on 21:00 UTC cutoff for FX, 22:00 UTC for ZN...")
    proc["trading_day_fx"] = proc["ts_article_utc"].apply(lambda x: assign_trading_day(x, 21))
    proc["trading_day_zn"] = proc["ts_article_utc"].apply(lambda x: assign_trading_day(x, 22))
    
    # Log trading day distribution for verification
    fx_days = proc["trading_day_fx"].dt.date.value_counts().sort_index()
    zn_days = proc["trading_day_zn"].dt.date.value_counts().sort_index()
    logger.info(f"FX trading days: {len(fx_days)} unique dates, range: {fx_days.index.min()} to {fx_days.index.max()}")
    logger.info(f"ZN trading days: {len(zn_days)} unique dates, range: {zn_days.index.min()} to {zn_days.index.max()}")
    
    # Sample verification
    sample_trading_days = proc[["ts_article_utc", "trading_day_fx", "trading_day_zn"]].head(3)
    logger.info(f"Sample trading day mapping:\n{sample_trading_days}")

    # 5) Save both CSV and Parquet (faster downstream) with robust fallback
    out_dir = Path("data/raw/gdelt")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV for compatibility (always works)
    csv_path = out_dir / "gdelt_macro_events_with_headlines.csv"
    proc.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")
    
    # Best-effort Parquet writer with fallback
    parq_path = out_dir / "gdelt_macro_events_with_headlines.parquet"
    csv_gz_path = out_dir / "gdelt_macro_events_with_headlines.csv.gz"
    
    try:
        import pyarrow  # noqa
        engine = "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa
            engine = "fastparquet"
        except Exception:
            engine = None
    
    if engine:
        proc.to_parquet(parq_path, engine=engine, index=False)
        logger.info(f"Saved Parquet to {parq_path} (engine={engine})")
    else:
        logger.warning("No parquet engine; falling back to gzipped CSV")
        proc.to_csv(csv_gz_path, index=False, compression="gzip")
        logger.info(f"Saved gzipped CSV to {csv_gz_path}")

if __name__ == "__main__":
    main()
