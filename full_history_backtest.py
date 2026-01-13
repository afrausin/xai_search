#!/usr/bin/env python3
"""
Full History Backtest

Run backtest on all 200 tickers (original + alternative) with maximum available history.
"""

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')

from high_return_optimization import precompute_indicators, high_conviction_backtest

# Original 100 tickers (large-cap)
ORIGINAL_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    'CRM', 'AMD', 'INTC', 'QCOM', 'TXN', 'CSCO', 'IBM', 'AMAT', 'MU', 'LRCX',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
    'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'HD', 'LOW',
    'CAT', 'BA', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'MMM', 'FDX',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'WBD', 'PARA',
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'PPG',
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'NEE', 'DUK', 'SO', 'D', 'AEP',
]

# Alternative 100 tickers (mid-cap/different sectors)
ALTERNATIVE_TICKERS = [
    'SNOW', 'DDOG', 'ZS', 'CRWD', 'NET', 'OKTA', 'TEAM', 'WDAY', 'SPLK', 'PANW',
    'FTNT', 'ZM', 'DOCU', 'TWLO', 'SQ', 'SHOP', 'ROKU', 'TTD', 'SNAP', 'PINS',
    'TFC', 'PNC', 'FITB', 'KEY', 'CFG', 'RF', 'HBAN', 'ALLY', 'SIVB', 'ZION',
    'VRTX', 'REGN', 'BIIB', 'GILD', 'ILMN', 'ALGN', 'IDXX', 'DXCM', 'ZBH', 'BSX',
    'TGT', 'DG', 'DLTR', 'ROST', 'TJX', 'ORLY', 'AZO', 'BBY', 'ULTA', 'LULU',
    'EMR', 'ETN', 'ROK', 'PH', 'ITW', 'CMI', 'PCAR', 'FAST', 'ODFL', 'CARR',
    'DVN', 'FANG', 'HAL', 'BKR', 'APA', 'MRO', 'CTRA', 'OVV', 'HES', 'TRGP',
    'MTCH', 'LYV', 'TTWO', 'FOXA', 'IPG', 'OMC', 'NWSA', 'VIAC', 'LBRDK', 'GOOG',
    'PYPL', 'FIS', 'FISV', 'GPN', 'VRSN', 'CDW', 'CTSH', 'EPAM', 'IT', 'AKAM',
    'MAR', 'HLT', 'LVS', 'WYNN', 'MGM', 'CCL', 'RCL', 'NCLH', 'DAL', 'UAL',
]

# Combine all 200 tickers
ALL_TICKERS = ORIGINAL_TICKERS + ALTERNATIVE_TICKERS

# Optimal balanced parameters from the optimization
OPTIMAL_PARAMS = {
    'z_threshold': 3.0,
    'max_hurst': 0.45,
    'min_volatility': 0.2,
    'min_duration': 2,
    'min_amp_ratio': 0.5,
    'min_exit': 7,
    'max_exit': 20,
    'ext_lb': 15,
    'hurst_lb': 25
}


def fetch_yahoo_data_extended(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance with extended history.

    Args:
        ticker: Stock symbol
        years: Number of years of history to fetch (default 10)

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    end = datetime.now()
    start = end - timedelta(days=years * 365)

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
        response = requests.get(url, params=params, headers=headers, timeout=15)
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


def download_all_extended(tickers: list, years: int = 10) -> dict:
    """Download extended history for all tickers with parallel processing."""
    data = {}
    print(f"Downloading {years} years of history for {len(tickers)} tickers...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_yahoo_data_extended, t, years): t for t in tickers}

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


def analyze_trades(trades: list, name: str) -> dict:
    """Analyze trades and return metrics."""
    if not trades:
        return None

    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']

    if not long_trades or not short_trades:
        return None

    long_beta = np.mean([t['beta_adj_return'] for t in long_trades]) * 100
    short_beta = np.mean([t['beta_adj_return'] for t in short_trades]) * 100
    long_raw = np.mean([t['return'] for t in long_trades]) * 100
    short_raw = np.mean([t['return'] for t in short_trades]) * 100

    long_hr = sum(1 for t in long_trades if t['beta_adj_win']) / len(long_trades) * 100
    short_hr = sum(1 for t in short_trades if t['beta_adj_win']) / len(short_trades) * 100

    return {
        'name': name,
        'total_trades': len(trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_beta': long_beta,
        'short_beta': short_beta,
        'avg_beta': (long_beta + short_beta) / 2,
        'imbalance': abs(long_beta - short_beta),
        'long_raw': long_raw,
        'short_raw': short_raw,
        'long_hr': long_hr,
        'short_hr': short_hr,
        'trades_list': trades
    }


def main():
    print("=" * 70)
    print("FULL HISTORY BACKTEST - 200 STOCKS")
    print("=" * 70)

    years = 10  # Fetch 10 years of history

    # Download SPY for beta calculation
    print(f"\nDownloading SPY ({years} years) for beta calculation...")
    market_df = fetch_yahoo_data_extended('SPY', years=years)
    if market_df is None:
        print("ERROR: Could not download SPY data")
        return
    print(f"SPY data: {len(market_df)} days from {market_df.index[0].date()} to {market_df.index[-1].date()}")

    # Download all 200 tickers
    print(f"\nDownloading all {len(ALL_TICKERS)} tickers...")
    data = download_all_extended(ALL_TICKERS, years=years)

    # Track which universe each ticker belongs to
    original_set = set(ORIGINAL_TICKERS)
    alternative_set = set(ALTERNATIVE_TICKERS)

    # Precompute indicators
    print("\nPrecomputing indicators...")
    indicators_cache = {}
    ticker_universe = {}  # Track which universe each ticker is from

    for ticker, df in tqdm(data.items(), desc="Computing indicators"):
        try:
            indicators_cache[ticker] = precompute_indicators(df, market_df)
            if ticker in original_set:
                ticker_universe[ticker] = 'original'
            else:
                ticker_universe[ticker] = 'alternative'
        except Exception as e:
            continue

    print(f"Precomputed for {len(indicators_cache)} tickers")

    original_count = sum(1 for t in indicators_cache if ticker_universe.get(t) == 'original')
    alt_count = sum(1 for t in indicators_cache if ticker_universe.get(t) == 'alternative')
    print(f"  Original: {original_count}, Alternative: {alt_count}")

    # Run backtest on all tickers
    print("\nRunning backtest...")
    all_trades = []
    original_trades = []
    alternative_trades = []

    for ticker, indicators in tqdm(indicators_cache.items(), desc="Backtesting"):
        trades = high_conviction_backtest(indicators, ticker, **OPTIMAL_PARAMS)
        all_trades.extend(trades)

        if ticker_universe.get(ticker) == 'original':
            original_trades.extend(trades)
        else:
            alternative_trades.extend(trades)

    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    all_results = analyze_trades(all_trades, "All 200 Tickers")
    orig_results = analyze_trades(original_trades, "Original 100")
    alt_results = analyze_trades(alternative_trades, "Alternative 100")

    for results in [all_results, orig_results, alt_results]:
        if results:
            print(f"\n{results['name']}:")
            print(f"  Total Trades: {results['total_trades']} (L:{results['long_trades']}, S:{results['short_trades']})")
            print(f"  Beta-Adj Return: L:{results['long_beta']:.2f}%, S:{results['short_beta']:.2f}%")
            print(f"  Avg Beta-Adj: {results['avg_beta']:.2f}% | Imbalance: {results['imbalance']:.2f}%")
            print(f"  Raw Return: L:{results['long_raw']:.2f}%, S:{results['short_raw']:.2f}%")
            print(f"  Hit Rate: L:{results['long_hr']:.1f}%, S:{results['short_hr']:.1f}%")

    # Breakdown by sector
    if all_results:
        print("\n" + "=" * 70)
        print("BREAKDOWN BY SECTOR")
        print("=" * 70)

        # Group trades by sector
        trades_by_sector = {}
        for t in all_trades:
            sector = t.get('sector', 'unknown')
            if sector not in trades_by_sector:
                trades_by_sector[sector] = []
            trades_by_sector[sector].append(t)

        print(f"\n{'Sector':<20} | {'Trades':>7} | {'L Beta%':>8} | {'S Beta%':>8} | {'Avg%':>8} | {'L HR%':>6} | {'S HR%':>6}")
        print("-" * 85)

        sector_results = []
        for sector in sorted(trades_by_sector.keys()):
            trades = trades_by_sector[sector]
            long_t = [t for t in trades if t['direction'] == 'long']
            short_t = [t for t in trades if t['direction'] == 'short']

            if long_t and short_t:
                l_beta = np.mean([t['beta_adj_return'] for t in long_t]) * 100
                s_beta = np.mean([t['beta_adj_return'] for t in short_t]) * 100
                l_hr = sum(1 for t in long_t if t['beta_adj_win']) / len(long_t) * 100
                s_hr = sum(1 for t in short_t if t['beta_adj_win']) / len(short_t) * 100
                avg = (l_beta + s_beta) / 2
                sector_results.append((sector, len(trades), l_beta, s_beta, avg, l_hr, s_hr))

        # Sort by avg performance
        sector_results.sort(key=lambda x: x[4], reverse=True)
        for sector, n, l_beta, s_beta, avg, l_hr, s_hr in sector_results:
            print(f"{sector:<20} | {n:>7} | {l_beta:>8.2f} | {s_beta:>8.2f} | {avg:>8.2f} | {l_hr:>6.1f} | {s_hr:>6.1f}")

    # Top tickers by performance
    print("\n" + "=" * 70)
    print("TOP PERFORMING TICKERS (by avg beta-adjusted return)")
    print("=" * 70)

    # Aggregate by ticker
    ticker_stats = {}
    for t in all_trades:
        ticker = t['ticker']
        if ticker not in ticker_stats:
            ticker_stats[ticker] = {'long': [], 'short': [], 'universe': ticker_universe.get(ticker, 'unknown')}
        ticker_stats[ticker][t['direction']].append(t['beta_adj_return'] * 100)

    # Calculate averages
    ticker_avgs = []
    for ticker, stats in ticker_stats.items():
        long_avg = np.mean(stats['long']) if stats['long'] else None
        short_avg = np.mean(stats['short']) if stats['short'] else None

        if long_avg is not None and short_avg is not None:
            avg = (long_avg + short_avg) / 2
            ticker_avgs.append({
                'ticker': ticker,
                'universe': stats['universe'],
                'long': long_avg,
                'short': short_avg,
                'avg': avg,
                'n_trades': len(stats['long']) + len(stats['short'])
            })

    # Sort by average
    ticker_avgs.sort(key=lambda x: x['avg'], reverse=True)

    print(f"\n{'Rank':<5} | {'Ticker':<8} | {'Universe':<12} | {'Trades':>6} | {'L Beta%':>8} | {'S Beta%':>8} | {'Avg%':>8}")
    print("-" * 75)

    for i, t in enumerate(ticker_avgs[:20], 1):
        print(f"{i:<5} | {t['ticker']:<8} | {t['universe']:<12} | {t['n_trades']:>6} | {t['long']:>8.2f} | {t['short']:>8.2f} | {t['avg']:>8.2f}")

    print("\n... Bottom 10:")
    for i, t in enumerate(ticker_avgs[-10:], len(ticker_avgs) - 9):
        print(f"{i:<5} | {t['ticker']:<8} | {t['universe']:<12} | {t['n_trades']:>6} | {t['long']:>8.2f} | {t['short']:>8.2f} | {t['avg']:>8.2f}")


if __name__ == "__main__":
    main()
