"""
Sensitivity Analysis for High Return Optimization
Shows average returns for longs/shorts across key parameter values
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS, fetch_yahoo_data
from high_return_optimization import precompute_indicators, high_conviction_backtest, SECTOR_MAP

# Best parameters from optimization
BEST_PARAMS = {
    'z_threshold': 3.0,
    'max_hurst': 0.30,
    'min_volatility': 0.25,
    'min_duration': 3,
    'min_amp_ratio': 1.5,
    'min_exit': 5,
    'max_exit': 20,
    'ext_lb': 20,
    'hurst_lb': 25,
}

# Parameter ranges for sensitivity analysis
PARAM_RANGES = {
    'z_threshold': [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'max_hurst': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    'min_volatility': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    'min_duration': [1, 2, 3, 5, 7, 10],
    'min_amp_ratio': [0.5, 1.0, 1.5, 2.0, 2.5],
    'max_exit': [5, 10, 15, 20, 25, 30],
}


def run_backtest(precomputed: Dict, params: Dict) -> Dict:
    """Run backtest with given parameters"""
    all_trades = []

    for ticker, indicators in precomputed.items():
        trades = high_conviction_backtest(indicators, ticker, **params)
        all_trades.extend(trades)

    long_trades = [t for t in all_trades if t['direction'] == 'long']
    short_trades = [t for t in all_trades if t['direction'] == 'short']

    result = {
        'total_trades': len(all_trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
    }

    if long_trades:
        long_rets = [t['return'] for t in long_trades]
        result['long_avg'] = np.mean(long_rets) * 100
        result['long_hr'] = sum(1 for t in long_trades if t['win']) / len(long_trades) * 100
    else:
        result['long_avg'] = 0
        result['long_hr'] = 0

    if short_trades:
        short_rets = [t['return'] for t in short_trades]
        result['short_avg'] = np.mean(short_rets) * 100
        result['short_hr'] = sum(1 for t in short_trades if t['win']) / len(short_trades) * 100
    else:
        result['short_avg'] = 0
        result['short_hr'] = 0

    if all_trades:
        result['total_avg'] = np.mean([t['return'] for t in all_trades]) * 100
    else:
        result['total_avg'] = 0

    return result


def main():
    print("=" * 80)
    print("SENSITIVITY ANALYSIS - Average Returns by Parameter")
    print("=" * 80)

    # Download data
    print("\nDownloading data...")
    data = download_all_data(TICKERS, period="2y")
    market_df = fetch_yahoo_data('SPY', period='2y')

    # Precompute indicators
    print("Precomputing indicators...")
    precomputed = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            precomputed[ticker] = precompute_indicators(df, market_df)
        except:
            continue

    print(f"\nAnalyzing {len(precomputed)} tickers\n")

    # Run sensitivity analysis for each parameter
    for param_name, param_values in PARAM_RANGES.items():
        print("\n" + "=" * 80)
        print(f"SENSITIVITY: {param_name}")
        print("=" * 80)

        results = []
        for value in param_values:
            # Create params with this value, others at best
            params = BEST_PARAMS.copy()
            params[param_name] = value

            # Ensure min_exit < max_exit
            if param_name == 'max_exit' and value <= params['min_exit']:
                continue

            result = run_backtest(precomputed, params)
            result['param_value'] = value
            results.append(result)

        # Create table
        print(f"\n{'Value':>8} | {'Long Avg':>10} | {'Short Avg':>10} | {'Total Avg':>10} | {'L Trades':>8} | {'S Trades':>8} | {'L HR%':>6} | {'S HR%':>6}")
        print("-" * 90)

        for r in results:
            # Highlight best/optimal value
            marker = " *" if r['param_value'] == BEST_PARAMS.get(param_name) else ""
            print(f"{r['param_value']:>8} | {r['long_avg']:>9.2f}% | {r['short_avg']:>9.2f}% | {r['total_avg']:>9.2f}% | {r['long_trades']:>8} | {r['short_trades']:>8} | {r['long_hr']:>5.1f}% | {r['short_hr']:>5.1f}%{marker}")

        print("\n* = optimal value from optimization")

    # Summary table
    print("\n" + "=" * 80)
    print("BASELINE (Optimal Parameters)")
    print("=" * 80)
    baseline = run_backtest(precomputed, BEST_PARAMS)
    print(f"\nLong Avg Return:  {baseline['long_avg']:.2f}%")
    print(f"Short Avg Return: {baseline['short_avg']:.2f}%")
    print(f"Total Avg Return: {baseline['total_avg']:.2f}%")
    print(f"\nTrades: {baseline['total_trades']} (L:{baseline['long_trades']}, S:{baseline['short_trades']})")
    print(f"Hit Rates: L:{baseline['long_hr']:.1f}%, S:{baseline['short_hr']:.1f}%")


if __name__ == "__main__":
    main()
