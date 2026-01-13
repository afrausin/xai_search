"""
Quick Market Neutral Test - Focused Parameter Set
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import warnings
import time
from multiprocessing import Pool, cpu_count
import itertools
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS, fetch_yahoo_data
from market_neutral_backtest import (
    precompute_indicators_with_market,
    market_neutral_backtest,
    aggregate_market_neutral_results
)


def test_config(args):
    """Test a single configuration"""
    precomputed, params = args
    all_trades = []

    for ticker, indicators in precomputed.items():
        result = market_neutral_backtest(indicators, **params)
        all_trades.extend(result['trades'])

    if len(all_trades) < 100:
        return None, None, 0

    result = aggregate_market_neutral_results(all_trades)

    if result['long_trades'] < 30 or result['short_trades'] < 30:
        return None, None, 0

    # Balance ratio
    trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])

    # Beta penalty
    beta_penalty = abs(result['net_beta_exposure']) / result['total'] if result['total'] > 0 else 1

    # Score
    avg_hr = (result['long_hit_rate'] + result['short_hit_rate']) / 2
    score = avg_hr * (0.5 + 0.5 * trade_ratio) * (1 - 0.1 * beta_penalty)

    return result, params, score


def main():
    """Quick market neutral test"""
    print("=" * 60)
    print("QUICK MARKET NEUTRAL TEST")
    print("=" * 60)

    # Download SPY
    print("\nDownloading SPY...")
    spy = fetch_yahoo_data("SPY", period="2y")
    if spy is None:
        print("Failed to download SPY")
        return

    spy_close = spy['Close'].values.flatten()
    market_returns = np.zeros(len(spy_close))
    market_returns[1:] = (spy_close[1:] - spy_close[:-1]) / spy_close[:-1]
    print(f"SPY: {len(spy_close)} days")

    # Download tickers
    data = download_all_data(TICKERS, period="2y")
    print(f"Downloaded {len(data)} tickers")

    # Precompute
    print("\nPrecomputing indicators...")
    precomputed = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            precomputed[ticker] = precompute_indicators_with_market(df, market_returns)
        except:
            continue
    print(f"Precomputed {len(precomputed)} tickers")

    # Focused parameter grid
    print("\nBuilding parameter grid...")
    configs = []

    # Test 3 strategies: balanced, long-biased high HR, short-biased high HR
    for base_thresh in [3.0, 3.2, 3.5, 3.8]:
        for max_hurst in [0.35, 0.40, 0.45]:
            for min_exit, max_exit in [(3, 6), (4, 7), (5, 8)]:
                for short_tm in [0.8, 0.9, 1.0, 1.1]:
                    for short_ha in [0.0, 0.02, 0.05]:
                        for short_md in [2, 3, 4, 5]:
                            for short_mar in [1.0, 1.2, 1.4]:
                                params = {
                                    'base_threshold': base_thresh,
                                    'ext_lb': 15,
                                    'hurst_lb': 20,
                                    'max_hurst': max_hurst,
                                    'min_exit': min_exit,
                                    'max_exit': max_exit,
                                    'inertia_sens': 0.8,
                                    'duration_weight': 0.05,
                                    'amplitude_weight': 0.5,
                                    'short_threshold_mult': short_tm,
                                    'short_max_hurst_adj': short_ha,
                                    'short_exit_mult': 1.0,
                                    'short_min_duration': short_md,
                                    'short_min_amp_ratio': short_mar,
                                    'target_vol': 0.15,
                                }
                                configs.append((precomputed, params))

    print(f"Testing {len(configs)} configurations with {cpu_count()} cores...")

    # Parallel execution
    best = None
    best_params = None
    best_score = 0

    batch_size = 200
    for batch_start in tqdm(range(0, len(configs), batch_size), desc="Batches"):
        batch = configs[batch_start:batch_start + batch_size]

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(test_config, batch)

        for result, params, score in results:
            if result is None:
                continue
            if score > best_score:
                best_score = score
                best = result
                best_params = params.copy()
                trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])
                print(f"\n  New best: Long {result['long_hit_rate']:.1f}% ({result['long_trades']}), "
                      f"Short {result['short_hit_rate']:.1f}% ({result['short_trades']}), "
                      f"Balance: {trade_ratio:.2f}, Beta: {result['net_beta_exposure']:.1f}")

    if best:
        trade_ratio = min(best['long_trades'], best['short_trades']) / max(best['long_trades'], best['short_trades'])
        print("\n" + "=" * 60)
        print("BEST MARKET NEUTRAL RESULT")
        print("=" * 60)
        print(f"Total Trades: {best['total']}")
        print(f"  Long: {best['long_trades']}, Short: {best['short_trades']}")
        print(f"  Trade Balance: {trade_ratio:.2f}")
        print(f"\nHit Rates:")
        print(f"  Long:  {best['long_hit_rate']:.2f}%")
        print(f"  Short: {best['short_hit_rate']:.2f}%")
        print(f"\nPnL:")
        print(f"  Long:  {best['long_pnl']:.4f}")
        print(f"  Short: {best['short_pnl']:.4f}")
        print(f"  Total: {best['total_pnl']:.4f}")
        print(f"\nBeta:")
        print(f"  Net Beta Exposure: {best['net_beta_exposure']:.2f}")
        print(f"  Beta per Trade: {best['net_beta_exposure']/best['total']:.4f}")
        print(f"\nParameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    return {'result': best, 'params': best_params}


if __name__ == "__main__":
    result = main()
