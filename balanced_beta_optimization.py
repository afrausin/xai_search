"""
Balanced Beta-Adjusted Optimization
Target: Similar beta-adjusted returns for BOTH longs AND shorts
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import warnings
import itertools
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS, fetch_yahoo_data
from high_return_optimization import precompute_indicators, high_conviction_backtest, SECTOR_MAP


# Global precomputed for multiprocessing
_precomputed = None


def init_worker(precomputed):
    global _precomputed
    _precomputed = precomputed


def test_config(params):
    """Test a single configuration - optimize for balanced beta-adjusted returns"""
    global _precomputed
    all_trades = []

    for ticker, indicators in _precomputed.items():
        trades = high_conviction_backtest(indicators, ticker, **params)
        all_trades.extend(trades)

    # Need minimum trades for statistical significance
    if len(all_trades) < 20:
        return None, None, -999

    long_trades = [t for t in all_trades if t['direction'] == 'long']
    short_trades = [t for t in all_trades if t['direction'] == 'short']

    # Need both longs and shorts
    if len(long_trades) < 5 or len(short_trades) < 5:
        return None, None, -999

    # Calculate beta-adjusted returns
    long_beta_adj = np.mean([t['beta_adj_return'] for t in long_trades]) * 100
    short_beta_adj = np.mean([t['beta_adj_return'] for t in short_trades]) * 100

    long_hr = sum(1 for t in long_trades if t['beta_adj_win']) / len(long_trades) * 100
    short_hr = sum(1 for t in short_trades if t['beta_adj_win']) / len(short_trades) * 100

    # Scoring function: Reward balanced returns
    # Use minimum of (long, short) to ensure both are good
    # Penalize large imbalance
    min_return = min(long_beta_adj, short_beta_adj)
    avg_return = (long_beta_adj + short_beta_adj) / 2
    imbalance = abs(long_beta_adj - short_beta_adj)

    # Score: average of min and avg, penalized by imbalance
    # This rewards configs where both sides are profitable AND similar
    score = (min_return + avg_return) / 2 - imbalance * 0.2

    # Bonus if both sides positive
    if long_beta_adj > 0 and short_beta_adj > 0:
        score += 1.0

    # Bonus for higher hit rates
    if long_hr > 55 and short_hr > 55:
        score += 0.5

    result = {
        'total_trades': len(all_trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_beta_adj': long_beta_adj,
        'short_beta_adj': short_beta_adj,
        'avg_beta_adj': avg_return,
        'min_beta_adj': min_return,
        'imbalance': imbalance,
        'long_hr': long_hr,
        'short_hr': short_hr,
        'long_raw': np.mean([t['return'] for t in long_trades]) * 100,
        'short_raw': np.mean([t['return'] for t in short_trades]) * 100,
    }

    return result, params, score


def main():
    print("=" * 70)
    print("BALANCED BETA-ADJUSTED OPTIMIZATION")
    print("Target: Similar returns for longs AND shorts")
    print("=" * 70)

    # Download data
    data = download_all_data(TICKERS, period="2y")
    print(f"\nDownloaded {len(data)} tickers")

    market_df = fetch_yahoo_data('SPY', period='2y')

    # Precompute
    print("\nPrecomputing indicators...")
    precomputed = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            precomputed[ticker] = precompute_indicators(df, market_df)
        except:
            continue

    print(f"Precomputed for {len(precomputed)} tickers")

    # Parameter grid - wider range to find balanced configs
    param_grid = {
        'z_threshold': [1.5, 2.0, 2.5, 3.0, 3.5],
        'max_hurst': [0.35, 0.40, 0.45, 0.50],
        'min_volatility': [0.10, 0.15, 0.20, 0.25],
        'min_duration': [1, 2, 3, 4, 5],
        'min_amp_ratio': [0.5, 0.75, 1.0, 1.25, 1.5],
        'min_exit': [3, 5, 7],
        'max_exit': [8, 12, 15, 20],
        'ext_lb': [15, 20],
        'hurst_lb': [20, 25],
    }

    # Generate configurations
    all_configs = []
    keys = list(param_grid.keys())
    for values in itertools.product(*[param_grid[k] for k in keys]):
        params = dict(zip(keys, values))
        if params['min_exit'] >= params['max_exit']:
            continue
        all_configs.append(params)

    print(f"\nTesting {len(all_configs)} configurations...")

    best_result = None
    best_params = None
    best_score = -999

    # Track top 10 for comparison
    top_results = []

    # Process in batches
    batch_size = 500
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(precomputed,)) as pool:
        for batch_start in tqdm(range(0, len(all_configs), batch_size), desc="Batches"):
            batch = all_configs[batch_start:batch_start + batch_size]
            results = pool.map(test_config, batch)

            for result, params, score in results:
                if result is None:
                    continue

                # Track top results
                top_results.append((score, result, params))
                top_results.sort(key=lambda x: -x[0])
                top_results = top_results[:10]

                if score > best_score:
                    best_score = score
                    best_result = result
                    best_params = params.copy()

                    print(f"\n  New best (score={score:.2f}):")
                    print(f"    Beta-Adj: L:{result['long_beta_adj']:.2f}%, S:{result['short_beta_adj']:.2f}%")
                    print(f"    Trades: {result['total_trades']} (L:{result['long_trades']}, S:{result['short_trades']})")
                    print(f"    Hit Rates: L:{result['long_hr']:.1f}%, S:{result['short_hr']:.1f}%")
                    print(f"    Imbalance: {result['imbalance']:.2f}%")

    # Print top 10 results
    print("\n" + "=" * 70)
    print("TOP 10 BALANCED CONFIGURATIONS")
    print("=" * 70)
    print(f"\n{'#':>2} | {'L Beta%':>8} | {'S Beta%':>8} | {'Imbal':>6} | {'L HR':>5} | {'S HR':>5} | {'L#':>4} | {'S#':>4} | {'Score':>6}")
    print("-" * 75)

    for i, (score, result, params) in enumerate(top_results, 1):
        print(f"{i:>2} | {result['long_beta_adj']:>7.2f}% | {result['short_beta_adj']:>7.2f}% | {result['imbalance']:>5.2f}% | {result['long_hr']:>4.1f}% | {result['short_hr']:>4.1f}% | {result['long_trades']:>4} | {result['short_trades']:>4} | {score:>6.2f}")

    if best_result:
        print("\n" + "=" * 70)
        print("BEST BALANCED RESULT")
        print("=" * 70)
        print(f"\nBeta-Adjusted Returns:")
        print(f"  Long:  {best_result['long_beta_adj']:.2f}%")
        print(f"  Short: {best_result['short_beta_adj']:.2f}%")
        print(f"  Avg:   {best_result['avg_beta_adj']:.2f}%")
        print(f"  Imbalance: {best_result['imbalance']:.2f}%")

        print(f"\nRaw Returns:")
        print(f"  Long:  {best_result['long_raw']:.2f}%")
        print(f"  Short: {best_result['short_raw']:.2f}%")

        print(f"\nTrades: {best_result['total_trades']} (L:{best_result['long_trades']}, S:{best_result['short_trades']})")
        print(f"Hit Rates: L:{best_result['long_hr']:.1f}%, S:{best_result['short_hr']:.1f}%")

        print(f"\nOptimal Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    return best_result, best_params, top_results


if __name__ == "__main__":
    main()
