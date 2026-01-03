"""
Simple data fetcher using Yahoo Finance API directly
"""

import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_yahoo_data(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from Yahoo Finance using their public API
    """
    # Calculate date range
    end = datetime.now()
    if period == "2y":
        start = end - timedelta(days=730)
    elif period == "1y":
        start = end - timedelta(days=365)
    else:
        start = end - timedelta(days=365)

    period1 = int(start.timestamp())
    period2 = int(end.timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
            return None

        result = data["chart"]["result"][0]

        timestamps = result.get("timestamp", [])
        if not timestamps:
            return None

        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit="s"),
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        })

        df = df.set_index("Date")
        df = df.dropna()

        if len(df) < 200:
            return None

        return df

    except Exception as e:
        return None


def download_all_data(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    """Download data for all tickers with parallel processing"""
    data = {}
    print(f"Downloading data for {len(tickers)} tickers...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_yahoo_data, t, period): t for t in tickers}

        for future in tqdm(as_completed(futures), total=len(tickers)):
            ticker = futures[future]
            try:
                df = future.result()
                if df is not None:
                    data[ticker] = df
                time.sleep(0.1)  # Rate limiting
            except Exception:
                pass

    print(f"Successfully downloaded {len(data)} tickers")
    return data


# Top 100 liquid stocks
TICKERS = [
    # Mega-cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    'CRM', 'AMD', 'INTC', 'QCOM', 'TXN', 'CSCO', 'IBM', 'AMAT', 'MU', 'LRCX',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'HD', 'LOW',
    # Industrial
    'CAT', 'BA', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'MMM', 'FDX',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
    # Communication
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'WBD', 'PARA',
    # Materials
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'PPG',
    # Real Estate & Utilities
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'NEE', 'DUK', 'SO', 'D', 'AEP',
]
