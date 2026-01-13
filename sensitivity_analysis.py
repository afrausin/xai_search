"""
Sensitivity Analysis for High Return Optimization
Shows average returns (raw, beta-adjusted, vol-adjusted) for longs/shorts
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
    'z_threshold': [2.0, 2.5, 3.0, 3.5, 4.0],
    'max_hurst': [0.25, 0.30, 0.35, 0.40, 0.45],
    'min_volatility': [0.10, 0.15, 0.20, 0.25, 0.30],
    'min_duration': [1, 2, 3, 5, 7],
    'min_amp_ratio': [0.5, 1.0, 1.5, 2.0],
    'max_exit': [10, 15, 20, 25, 30],
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

    # Raw returns
    if long_trades:
        result['long_raw'] = np.mean([t['return'] for t in long_trades]) * 100
        result['long_beta_adj'] = np.mean([t['beta_adj_return'] for t in long_trades]) * 100
        result['long_vol_adj'] = np.mean([t['vol_adj_return'] for t in long_trades]) * 100
        result['long_hr'] = sum(1 for t in long_trades if t['win']) / len(long_trades) * 100
    else:
        result['long_raw'] = result['long_beta_adj'] = result['long_vol_adj'] = 0
        result['long_hr'] = 0

    if short_trades:
        result['short_raw'] = np.mean([t['return'] for t in short_trades]) * 100
        result['short_beta_adj'] = np.mean([t['beta_adj_return'] for t in short_trades]) * 100
        result['short_vol_adj'] = np.mean([t['vol_adj_return'] for t in short_trades]) * 100
        result['short_hr'] = sum(1 for t in short_trades if t['win']) / len(short_trades) * 100
    else:
        result['short_raw'] = result['short_beta_adj'] = result['short_vol_adj'] = 0
        result['short_hr'] = 0

    # Combined
    if all_trades:
        result['total_raw'] = np.mean([t['return'] for t in all_trades]) * 100
        result['total_beta_adj'] = np.mean([t['beta_adj_return'] for t in all_trades]) * 100
        result['total_vol_adj'] = np.mean([t['vol_adj_return'] for t in all_trades]) * 100
    else:
        result['total_raw'] = result['total_beta_adj'] = result['total_vol_adj'] = 0

    return result


def print_table(results: List[Dict], param_name: str, best_value):
    """Print formatted sensitivity table"""

    # Table 1: Raw Returns
    print(f"\n  RAW RETURNS:")
    print(f"  {'Value':>8} | {'Long':>8} | {'Short':>8} | {'Total':>8} | {'L#':>4} | {'S#':>4}")
    print("  " + "-" * 55)
    for r in results:
        marker = "*" if r['param_value'] == best_value else " "
        print(f" {marker}{r['param_value']:>7} | {r['long_raw']:>7.2f}% | {r['short_raw']:>7.2f}% | {r['total_raw']:>7.2f}% | {r['long_trades']:>4} | {r['short_trades']:>4}")

    # Table 2: Beta-Adjusted Returns
    print(f"\n  BETA-ADJUSTED RETURNS:")
    print(f"  {'Value':>8} | {'Long':>8} | {'Short':>8} | {'Total':>8} | {'L HR':>6} | {'S HR':>6}")
    print("  " + "-" * 60)
    for r in results:
        marker = "*" if r['param_value'] == best_value else " "
        print(f" {marker}{r['param_value']:>7} | {r['long_beta_adj']:>7.2f}% | {r['short_beta_adj']:>7.2f}% | {r['total_beta_adj']:>7.2f}% | {r['long_hr']:>5.1f}% | {r['short_hr']:>5.1f}%")

    # Table 3: Vol-Adjusted Returns
    print(f"\n  VOL-ADJUSTED RETURNS (15% target):")
    print(f"  {'Value':>8} | {'Long':>8} | {'Short':>8} | {'Total':>8} | {'L HR':>6} | {'S HR':>6}")
    print("  " + "-" * 60)
    for r in results:
        marker = "*" if r['param_value'] == best_value else " "
        print(f" {marker}{r['param_value']:>7} | {r['long_vol_adj']:>7.2f}% | {r['short_vol_adj']:>7.2f}% | {r['total_vol_adj']:>7.2f}% | {r['long_hr']:>5.1f}% | {r['short_hr']:>5.1f}%")


def main():
    print("=" * 80)
    print("SENSITIVITY ANALYSIS - Beta & Vol Adjusted Returns")
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
            params = BEST_PARAMS.copy()
            params[param_name] = value

            if param_name == 'max_exit' and value <= params['min_exit']:
                continue

            result = run_backtest(precomputed, params)
            result['param_value'] = value
            results.append(result)

        print_table(results, param_name, BEST_PARAMS.get(param_name))
        print("\n  * = optimal value from prior optimization")

    # Summary with baseline
    print("\n" + "=" * 80)
    print("BASELINE (Optimal Parameters)")
    print("=" * 80)
    baseline = run_backtest(precomputed, BEST_PARAMS)

    print(f"\n  {'':15} | {'Raw':>10} | {'Beta-Adj':>10} | {'Vol-Adj':>10}")
    print("  " + "-" * 52)
    print(f"  {'Long Avg':15} | {baseline['long_raw']:>9.2f}% | {baseline['long_beta_adj']:>9.2f}% | {baseline['long_vol_adj']:>9.2f}%")
    print(f"  {'Short Avg':15} | {baseline['short_raw']:>9.2f}% | {baseline['short_beta_adj']:>9.2f}% | {baseline['short_vol_adj']:>9.2f}%")
    print(f"  {'Total Avg':15} | {baseline['total_raw']:>9.2f}% | {baseline['total_beta_adj']:>9.2f}% | {baseline['total_vol_adj']:>9.2f}%")
    print(f"\n  Trades: {baseline['total_trades']} (L:{baseline['long_trades']}, S:{baseline['short_trades']})")
    print(f"  Hit Rates: L:{baseline['long_hr']:.1f}%, S:{baseline['short_hr']:.1f}%")


if __name__ == "__main__":
    main()
