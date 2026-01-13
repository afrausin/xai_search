#!/usr/bin/env python3
"""
Ticker Universe Comparison

Test the strategy on 100 DIFFERENT tickers to validate robustness.
Compare performance against the original 100 tickers.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS as ORIGINAL_TICKERS, fetch_yahoo_data
from high_return_optimization import precompute_indicators, high_conviction_backtest

# 100 DIFFERENT tickers (mid-cap, smaller large-cap, different sectors)
# Explicitly excluding all tickers from ORIGINAL_TICKERS
ALTERNATIVE_TICKERS = [
    # Mid-cap Tech
    'SNOW', 'DDOG', 'ZS', 'CRWD', 'NET', 'OKTA', 'TEAM', 'WDAY', 'SPLK', 'PANW',
    'FTNT', 'ZM', 'DOCU', 'TWLO', 'SQ', 'SHOP', 'ROKU', 'TTD', 'SNAP', 'PINS',
    # Financials (different from original)
    'TFC', 'PNC', 'FITB', 'KEY', 'CFG', 'RF', 'HBAN', 'ALLY', 'SIVB', 'ZION',
    # Healthcare (different)
    'VRTX', 'REGN', 'BIIB', 'GILD', 'ILMN', 'ALGN', 'IDXX', 'DXCM', 'ZBH', 'BSX',
    # Consumer (different)
    'TGT', 'DG', 'DLTR', 'ROST', 'TJX', 'ORLY', 'AZO', 'BBY', 'ULTA', 'LULU',
    # Industrial (different)
    'EMR', 'ETN', 'ROK', 'PH', 'ITW', 'CMI', 'PCAR', 'FAST', 'ODFL', 'CARR',
    # Energy (different)
    'DVN', 'FANG', 'HAL', 'BKR', 'APA', 'MRO', 'CTRA', 'OVV', 'HES', 'TRGP',
    # Communication/Media (different)
    'MTCH', 'LYV', 'TTWO', 'FOXA', 'IPG', 'OMC', 'NWSA', 'VIAC', 'LBRDK', 'GOOG',
    # Specialty/Other
    'PYPL', 'FIS', 'FISV', 'GPN', 'VRSN', 'CDW', 'CTSH', 'EPAM', 'IT', 'AKAM',
    # More diverse sectors
    'MAR', 'HLT', 'LVS', 'WYNN', 'MGM', 'CCL', 'RCL', 'NCLH', 'DAL', 'UAL',
]

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


def run_backtest_on_universe(tickers, name, market_df):
    """Run backtest on a given ticker universe and return results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    # Download data
    print(f"Downloading {len(tickers)} tickers...")
    data = download_all_data(tickers, period="2y")
    print(f"Successfully downloaded {len(data)} tickers")

    # Precompute indicators
    print("Precomputing indicators...")
    indicators_cache = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            indicators_cache[ticker] = precompute_indicators(df, market_df)
        except:
            continue
    print(f"Precomputed for {len(indicators_cache)} tickers")

    # Run backtest
    all_trades = []
    for ticker, indicators in indicators_cache.items():
        trades = high_conviction_backtest(indicators, ticker, **OPTIMAL_PARAMS)
        all_trades.extend(trades)

    if not all_trades:
        return None

    long_trades = [t for t in all_trades if t['direction'] == 'long']
    short_trades = [t for t in all_trades if t['direction'] == 'short']

    if not long_trades or not short_trades:
        return None

    # Calculate metrics
    long_beta = np.mean([t['beta_adj_return'] for t in long_trades]) * 100
    short_beta = np.mean([t['beta_adj_return'] for t in short_trades]) * 100
    long_raw = np.mean([t['return'] for t in long_trades]) * 100
    short_raw = np.mean([t['return'] for t in short_trades]) * 100

    long_hr = sum(1 for t in long_trades if t['beta_adj_win']) / len(long_trades) * 100
    short_hr = sum(1 for t in short_trades if t['beta_adj_win']) / len(short_trades) * 100

    return {
        'name': name,
        'tickers_available': len(indicators_cache),
        'total_trades': len(all_trades),
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
        'trades_list': all_trades
    }


def main():
    print("=" * 70)
    print("TICKER UNIVERSE COMPARISON")
    print("Testing strategy robustness across different stock universes")
    print("=" * 70)

    # Download market data once
    print("\nDownloading SPY for beta calculation...")
    market_df = fetch_yahoo_data('SPY', period='2y')

    # Run on original tickers
    original_results = run_backtest_on_universe(
        ORIGINAL_TICKERS,
        "Original 100 Tickers (Large-Cap)",
        market_df
    )

    # Run on alternative tickers
    alt_results = run_backtest_on_universe(
        ALTERNATIVE_TICKERS,
        "Alternative 100 Tickers (Mid-Cap/Different)",
        market_df
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    if original_results and alt_results:
        print(f"\n{'Metric':<25} | {'Original':>15} | {'Alternative':>15} | {'Diff':>10}")
        print("-" * 75)

        metrics = [
            ('Tickers Available', 'tickers_available', '', 0),
            ('Total Trades', 'total_trades', '', 0),
            ('Long Trades', 'long_trades', '', 0),
            ('Short Trades', 'short_trades', '', 0),
            ('Long Beta-Adj %', 'long_beta', '%', 2),
            ('Short Beta-Adj %', 'short_beta', '%', 2),
            ('Avg Beta-Adj %', 'avg_beta', '%', 2),
            ('Imbalance %', 'imbalance', '%', 2),
            ('Long Raw %', 'long_raw', '%', 2),
            ('Short Raw %', 'short_raw', '%', 2),
            ('Long Hit Rate %', 'long_hr', '%', 1),
            ('Short Hit Rate %', 'short_hr', '%', 1),
        ]

        for label, key, suffix, decimals in metrics:
            orig = original_results[key]
            alt = alt_results[key]
            diff = alt - orig

            if decimals == 0:
                print(f"{label:<25} | {orig:>14} | {alt:>14} | {diff:>+9}")
            else:
                fmt = f"{{:>{14-len(suffix)}.{decimals}f}}{suffix}"
                diff_fmt = f"{{:>+{9-len(suffix)}.{decimals}f}}{suffix}"
                print(f"{label:<25} | {fmt.format(orig)} | {fmt.format(alt)} | {diff_fmt.format(diff)}")

        # Performance summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        beta_diff_pct = (alt_results['avg_beta'] / original_results['avg_beta'] - 1) * 100
        trade_diff_pct = (alt_results['total_trades'] / original_results['total_trades'] - 1) * 100

        print(f"\nOriginal Universe:")
        print(f"  Beta-Adj Return: L:{original_results['long_beta']:.2f}%, S:{original_results['short_beta']:.2f}%")
        print(f"  Avg: {original_results['avg_beta']:.2f}% | Trades: {original_results['total_trades']}")

        print(f"\nAlternative Universe:")
        print(f"  Beta-Adj Return: L:{alt_results['long_beta']:.2f}%, S:{alt_results['short_beta']:.2f}%")
        print(f"  Avg: {alt_results['avg_beta']:.2f}% | Trades: {alt_results['total_trades']}")

        print(f"\nDifference:")
        print(f"  Avg Beta Return: {beta_diff_pct:+.1f}% {'better' if beta_diff_pct > 0 else 'worse'}")
        print(f"  Trade Count: {trade_diff_pct:+.1f}%")

        # Analyze by direction
        print("\n" + "=" * 70)
        print("DETAILED BREAKDOWN")
        print("=" * 70)

        for results in [original_results, alt_results]:
            print(f"\n{results['name']}:")
            long_trades = [t for t in results['trades_list'] if t['direction'] == 'long']
            short_trades = [t for t in results['trades_list'] if t['direction'] == 'short']

            # By ticker
            long_by_ticker = {}
            short_by_ticker = {}
            for t in long_trades:
                if t['ticker'] not in long_by_ticker:
                    long_by_ticker[t['ticker']] = []
                long_by_ticker[t['ticker']].append(t['beta_adj_return'] * 100)
            for t in short_trades:
                if t['ticker'] not in short_by_ticker:
                    short_by_ticker[t['ticker']] = []
                short_by_ticker[t['ticker']].append(t['beta_adj_return'] * 100)

            print(f"  Tickers with long signals: {len(long_by_ticker)}")
            print(f"  Tickers with short signals: {len(short_by_ticker)}")

            # Top performers
            if long_by_ticker:
                top_long = sorted([(k, np.mean(v)) for k, v in long_by_ticker.items()],
                                  key=lambda x: -x[1])[:5]
                print(f"  Top 5 long tickers: {', '.join([f'{t}({r:.1f}%)' for t, r in top_long])}")
            if short_by_ticker:
                top_short = sorted([(k, np.mean(v)) for k, v in short_by_ticker.items()],
                                   key=lambda x: -x[1])[:5]
                print(f"  Top 5 short tickers: {', '.join([f'{t}({r:.1f}%)' for t, r in top_short])}")


if __name__ == "__main__":
    main()
