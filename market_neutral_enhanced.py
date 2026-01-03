"""
Enhanced Market Neutral Optimization
Focus on improving minimum hit rate while maintaining neutrality
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
    aggregate_market_neutral_results
)
from hurst_signal import (calculate_hurst, calculate_zscore,
                          calculate_atr, calculate_formation_duration, calculate_wave_amplitude)


def enhanced_market_neutral_backtest(indicators: Dict, base_threshold: float, ext_lb: int,
                                     hurst_lb: int, max_hurst: float, min_exit: int, max_exit: int,
                                     inertia_sens: float, duration_weight: float = 0.1,
                                     amplitude_weight: float = 0.5,
                                     short_threshold_mult: float = 1.0,
                                     short_max_hurst_adj: float = 0.0,
                                     short_exit_mult: float = 1.0,
                                     short_min_duration: int = 0,
                                     short_min_amp_ratio: float = 1.0,
                                     target_vol: float = 0.15,
                                     # New parameters
                                     short_min_beta: float = 0.0,
                                     short_max_beta: float = 3.0,
                                     short_vol_filter: float = 0.0,
                                     use_fast_short_exit: bool = False) -> Dict:
    """
    Enhanced market neutral backtest with additional filters for shorts

    New parameters:
    - short_min_beta: Minimum beta for short signals (high beta = faster mean reversion)
    - short_max_beta: Maximum beta for short signals
    - short_vol_filter: Minimum volatility for short signals
    - use_fast_short_exit: Use faster exit for shorts (exit at first pullback)
    """
    close = indicators['close']
    n = indicators['n']
    zscore = indicators['zscores'].get(ext_lb, indicators['zscores'][15])
    hurst = indicators['hursts'].get(hurst_lb, indicators['hursts'][20])
    atr_ratio = indicators['atr_ratio']
    duration = indicators['durations'].get(ext_lb, indicators['durations'][15])
    wave_amp = indicators['wave_amplitude']
    beta = indicators['beta']
    volatility = indicators['volatility']

    trades = []

    min_idx = max(hurst_lb, ext_lb, 60, 25)
    base_bars = (min_exit + max_exit) / 2

    short_max_hurst = max_hurst + short_max_hurst_adj
    wave_amp_mean = np.mean(wave_amp[wave_amp > 0]) if np.any(wave_amp > 0) else 1.0

    for i in range(min_idx, n - max_exit):
        h = hurst[i]
        z = zscore[i]
        vol_adj = atr_ratio[i]
        dur = duration[i]
        amp = wave_amp[i]
        b = beta[i]
        vol = volatility[i]

        if np.isnan(vol) or vol <= 0 or np.isnan(b):
            continue

        position_size = min(target_vol / vol, 2.0)

        hurst_factor = 0.5 + h * inertia_sens
        base_threshold_adj = base_threshold * hurst_factor * (0.8 + vol_adj * 0.4)

        duration_factor = 1.0 + dur * duration_weight
        amplitude_factor = 1.0 + (amp / wave_amp_mean - 1.0) * amplitude_weight if wave_amp_mean > 0 else 1.0
        base_exit_bars = base_bars * hurst_factor * duration_factor * amplitude_factor

        # Long signal
        if h < max_hurst and z < -base_threshold_adj:
            exit_bars = int(np.clip(base_exit_bars, min_exit, max_exit))
            if i + exit_bars >= n:
                continue
            entry = close[i]
            exit_price = close[i + exit_bars]
            ret = (exit_price - entry) / entry

            trades.append({
                'direction': 'long',
                'entry_idx': i,
                'exit_idx': i + exit_bars,
                'return': ret,
                'position_size': position_size,
                'beta': b,
                'beta_exposure': position_size * b,
                'win': ret > 0,
            })

        # Short signal with enhanced filters
        elif h < short_max_hurst and z > base_threshold_adj * short_threshold_mult:
            amp_ratio = amp / wave_amp_mean if wave_amp_mean > 0 else 1.0

            # Basic inertia filters
            if dur < short_min_duration or amp_ratio < short_min_amp_ratio:
                continue

            # Beta filter for shorts (high beta stocks mean-revert faster)
            if b < short_min_beta or b > short_max_beta:
                continue

            # Volatility filter for shorts
            if vol < short_vol_filter:
                continue

            # Calculate exit bars
            short_exit_bars = int(np.clip(base_exit_bars * short_exit_mult, min_exit, max_exit))

            # Fast exit option: use shorter hold period for shorts
            if use_fast_short_exit:
                short_exit_bars = max(min_exit, short_exit_bars - 1)

            if i + short_exit_bars >= n:
                continue

            entry = close[i]
            exit_price = close[i + short_exit_bars]
            ret = (entry - exit_price) / entry

            trades.append({
                'direction': 'short',
                'entry_idx': i,
                'exit_idx': i + short_exit_bars,
                'return': ret,
                'position_size': position_size,
                'beta': b,
                'beta_exposure': -position_size * b,
                'win': ret > 0,
            })

    return {'trades': trades}


def test_config(args):
    """Test a single configuration"""
    precomputed, params = args
    all_trades = []

    for ticker, indicators in precomputed.items():
        result = enhanced_market_neutral_backtest(indicators, **params)
        all_trades.extend(result['trades'])

    if len(all_trades) < 100:
        return None, None, 0

    result = aggregate_market_neutral_results(all_trades)

    if result['long_trades'] < 30 or result['short_trades'] < 30:
        return None, None, 0

    trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])
    beta_penalty = abs(result['net_beta_exposure']) / result['total'] if result['total'] > 0 else 1

    # New scoring: prioritize MINIMUM hit rate
    min_hr = min(result['long_hit_rate'], result['short_hit_rate'])
    avg_hr = (result['long_hit_rate'] + result['short_hit_rate']) / 2

    # Heavily weight minimum hit rate
    # Score = min_hr * 0.7 + avg_hr * 0.3, adjusted for balance and beta
    score = (min_hr * 0.7 + avg_hr * 0.3) * (0.5 + 0.5 * trade_ratio) * (1 - 0.05 * beta_penalty)

    return result, params, score


def main():
    """Enhanced market neutral optimization"""
    print("=" * 60)
    print("ENHANCED MARKET NEUTRAL OPTIMIZATION")
    print("Focus: Maximize MINIMUM hit rate")
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

    # Enhanced parameter grid
    print("\nBuilding enhanced parameter grid...")
    configs = []

    # Core parameters
    for base_thresh in [3.0, 3.2, 3.5, 3.8]:
        for max_hurst in [0.35, 0.40, 0.45]:
            for min_exit, max_exit in [(3, 5), (3, 6), (4, 6), (4, 7)]:
                # Short parameters - focus on high-quality setups
                for short_tm in [1.0, 1.1, 1.2, 1.3]:  # Higher threshold for shorts
                    for short_ha in [-0.05, -0.02, 0.0]:  # Stricter Hurst for shorts
                        for short_em in [0.6, 0.8, 1.0]:  # Faster exits for shorts
                            for short_md in [3, 4, 5, 6]:  # Higher min duration
                                for short_mar in [1.2, 1.4, 1.6]:  # Higher amplitude
                                    # Beta filter for shorts
                                    for short_min_beta in [0.8, 1.0, 1.2]:  # Require high-beta for shorts
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
                                            'short_exit_mult': short_em,
                                            'short_min_duration': short_md,
                                            'short_min_amp_ratio': short_mar,
                                            'target_vol': 0.15,
                                            'short_min_beta': short_min_beta,
                                            'short_max_beta': 3.0,
                                            'short_vol_filter': 0.0,
                                            'use_fast_short_exit': short_em < 0.8,
                                        }
                                        configs.append((precomputed, params))

    print(f"Testing {len(configs)} configurations with {cpu_count()} cores...")

    # Parallel execution
    best = None
    best_params = None
    best_score = 0
    best_min_hr = 0

    batch_size = 200
    for batch_start in tqdm(range(0, len(configs), batch_size), desc="Batches"):
        batch = configs[batch_start:batch_start + batch_size]

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(test_config, batch)

        for result, params, score in results:
            if result is None:
                continue

            min_hr = min(result['long_hit_rate'], result['short_hit_rate'])

            # Only consider if minimum HR is improving
            if min_hr > best_min_hr or (min_hr == best_min_hr and score > best_score):
                best_min_hr = min_hr
                best_score = score
                best = result
                best_params = params.copy()
                trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])
                print(f"\n  New best: L:{result['long_hit_rate']:.1f}% S:{result['short_hit_rate']:.1f}% "
                      f"Min:{min_hr:.1f}% ({result['long_trades']}L/{result['short_trades']}S) "
                      f"Bal:{trade_ratio:.2f} Beta:{result['net_beta_exposure']:.1f}")

    if best:
        trade_ratio = min(best['long_trades'], best['short_trades']) / max(best['long_trades'], best['short_trades'])
        min_hr = min(best['long_hit_rate'], best['short_hit_rate'])

        print("\n" + "=" * 60)
        print("BEST ENHANCED MARKET NEUTRAL RESULT")
        print("=" * 60)
        print(f"Total Trades: {best['total']}")
        print(f"  Long: {best['long_trades']}, Short: {best['short_trades']}")
        print(f"  Trade Balance: {trade_ratio:.2f}")
        print(f"\nHit Rates:")
        print(f"  Long:    {best['long_hit_rate']:.2f}%")
        print(f"  Short:   {best['short_hit_rate']:.2f}%")
        print(f"  Minimum: {min_hr:.2f}%")
        print(f"\nPnL:")
        print(f"  Long:  {best['long_pnl']:.4f}")
        print(f"  Short: {best['short_pnl']:.4f}")
        print(f"  Total: {best['total_pnl']:.4f}")
        print(f"\nBeta:")
        print(f"  Net Beta Exposure: {best['net_beta_exposure']:.2f}")
        print(f"  Beta per Trade: {best['net_beta_exposure']/best['total']:.4f}")
        print(f"\nKey Parameters:")
        for k in ['short_threshold_mult', 'short_max_hurst_adj', 'short_exit_mult',
                  'short_min_duration', 'short_min_amp_ratio', 'short_min_beta']:
            print(f"  {k}: {best_params[k]}")

    return {'result': best, 'params': best_params}


if __name__ == "__main__":
    result = main()
