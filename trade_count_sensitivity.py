#!/usr/bin/env python3
"""
Trade Count Sensitivity Analysis

Shows the trade-off between number of trades and beta-adjusted returns.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS, fetch_yahoo_data
from high_return_optimization import precompute_indicators, high_conviction_backtest


def run_sensitivity():
    """Run sensitivity analysis varying parameters that affect trade count."""

    print("=" * 70)
    print("TRADE COUNT SENSITIVITY ANALYSIS")
    print("How does performance change as we increase number of trades?")
    print("=" * 70)

    # Download data
    print("\nDownloading data for 100 tickers...")
    data = download_all_data(TICKERS, period="2y")
    print(f"\nDownloaded {len(data)} tickers")

    # Download market data
    market_df = fetch_yahoo_data('SPY', period='2y')

    # Precompute indicators
    print("\nPrecomputing indicators...")
    indicators_cache = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            indicators_cache[ticker] = precompute_indicators(df, market_df)
        except:
            continue
    print(f"Precomputed for {len(indicators_cache)} tickers\n")

    # Baseline (optimal balanced params)
    baseline = {
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

    # Parameter variations to increase trade count
    # Each row: (param_name, values_from_strict_to_loose)
    param_sweeps = [
        ('z_threshold', [3.5, 3.0, 2.5, 2.0, 1.75, 1.5]),
        ('max_hurst', [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]),
        ('min_volatility', [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]),
        ('min_amp_ratio', [1.0, 0.75, 0.5, 0.3, 0.2, 0.1]),
    ]

    def test_params(params):
        """Run backtest and return results."""
        all_trades = []
        for ticker, indicators in indicators_cache.items():
            trades = high_conviction_backtest(indicators, ticker, **params)
            all_trades.extend(trades)

        if not all_trades:
            return None

        long_trades = [t for t in all_trades if t['direction'] == 'long']
        short_trades = [t for t in all_trades if t['direction'] == 'short']

        if not long_trades or not short_trades:
            return None

        long_beta = np.mean([t['beta_adj_return'] for t in long_trades]) * 100
        short_beta = np.mean([t['beta_adj_return'] for t in short_trades]) * 100
        avg_beta = (long_beta + short_beta) / 2

        long_hr = sum(1 for t in long_trades if t['beta_adj_win']) / len(long_trades) * 100
        short_hr = sum(1 for t in short_trades if t['beta_adj_win']) / len(short_trades) * 100

        return {
            'trades': len(all_trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_beta': long_beta,
            'short_beta': short_beta,
            'avg_beta': avg_beta,
            'imbalance': abs(long_beta - short_beta),
            'long_hr': long_hr,
            'short_hr': short_hr
        }

    # Test baseline first
    print("Testing baseline configuration...")
    baseline_result = test_params(baseline)
    print(f"  Baseline: {baseline_result['trades']} trades, "
          f"L:{baseline_result['long_beta']:.2f}%, S:{baseline_result['short_beta']:.2f}%\n")

    # Run sensitivity for each parameter
    for param_name, values in param_sweeps:
        print("=" * 70)
        print(f"SENSITIVITY: {param_name}")
        print("=" * 70)
        print(f"\n{'Value':>8} | {'Trades':>7} | {'L Beta%':>8} | {'S Beta%':>8} | {'Avg%':>7} | {'Imbal':>6} | {'L HR':>6} | {'S HR':>6}")
        print("-" * 80)

        for val in values:
            params = baseline.copy()
            params[param_name] = val

            result = test_params(params)
            if result:
                marker = " <-- baseline" if val == baseline[param_name] else ""
                print(f"{val:>8.2f} | {result['trades']:>7} | "
                      f"{result['long_beta']:>7.2f}% | {result['short_beta']:>7.2f}% | "
                      f"{result['avg_beta']:>6.2f}% | {result['imbalance']:>5.2f}% | "
                      f"{result['long_hr']:>5.1f}% | {result['short_hr']:>5.1f}%{marker}")
            else:
                print(f"{val:>8.2f} | {'N/A':>7} | {'N/A':>8} | {'N/A':>8} | {'N/A':>7}")

        print()

    # Combined sensitivity - gradually relax all parameters together
    print("=" * 70)
    print("COMBINED SENSITIVITY: Relaxing All Parameters Together")
    print("=" * 70)

    # Define relaxation levels (0 = baseline, 1 = most relaxed)
    relaxation_configs = [
        {"name": "Tight (baseline)", "z": 3.0, "h": 0.45, "v": 0.20, "a": 0.5},
        {"name": "Slightly loose", "z": 2.75, "h": 0.47, "v": 0.18, "a": 0.4},
        {"name": "Moderate", "z": 2.5, "h": 0.50, "v": 0.15, "a": 0.3},
        {"name": "Loose", "z": 2.25, "h": 0.52, "v": 0.12, "a": 0.25},
        {"name": "Very loose", "z": 2.0, "h": 0.55, "v": 0.10, "a": 0.2},
        {"name": "Ultra loose", "z": 1.75, "h": 0.58, "v": 0.08, "a": 0.15},
        {"name": "Maximum", "z": 1.5, "h": 0.60, "v": 0.05, "a": 0.1},
    ]

    print(f"\n{'Config':>18} | {'Trades':>7} | {'L Beta%':>8} | {'S Beta%':>8} | {'Avg%':>7} | {'Imbal':>6} | {'L HR':>6} | {'S HR':>6}")
    print("-" * 95)

    results_list = []
    for config in relaxation_configs:
        params = baseline.copy()
        params['z_threshold'] = config['z']
        params['max_hurst'] = config['h']
        params['min_volatility'] = config['v']
        params['min_amp_ratio'] = config['a']

        result = test_params(params)
        if result:
            results_list.append({**config, **result})
            print(f"{config['name']:>18} | {result['trades']:>7} | "
                  f"{result['long_beta']:>7.2f}% | {result['short_beta']:>7.2f}% | "
                  f"{result['avg_beta']:>6.2f}% | {result['imbalance']:>5.2f}% | "
                  f"{result['long_hr']:>5.1f}% | {result['short_hr']:>5.1f}%")

    # Show trade-off summary
    print("\n" + "=" * 70)
    print("TRADE-OFF SUMMARY")
    print("=" * 70)

    if len(results_list) >= 2:
        baseline_r = results_list[0]
        for r in results_list[1:]:
            trade_increase = r['trades'] - baseline_r['trades']
            trade_pct = (r['trades'] / baseline_r['trades'] - 1) * 100
            perf_loss = baseline_r['avg_beta'] - r['avg_beta']
            perf_pct = (1 - r['avg_beta'] / baseline_r['avg_beta']) * 100

            print(f"\n{r['name']}:")
            print(f"  Trades: {baseline_r['trades']} → {r['trades']} (+{trade_increase}, +{trade_pct:.0f}%)")
            print(f"  Avg Beta Return: {baseline_r['avg_beta']:.2f}% → {r['avg_beta']:.2f}% ({-perf_loss:+.2f}%, {-perf_pct:+.1f}%)")
            print(f"  Return per additional trade: {-perf_loss/(trade_increase+0.001):.3f}% lost per trade")


if __name__ == "__main__":
    run_sensitivity()
