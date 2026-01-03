"""
Targeted Short Optimization
Focus: Only short in sectors where it works (Energy, Materials, Healthcare, REITs, Utilities)
with stricter inertia requirements
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

# Sector mapping
SECTOR_MAP = {
    # Tech
    'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech', 'NVDA': 'tech',
    'META': 'tech', 'TSLA': 'tech', 'AVGO': 'tech', 'ORCL': 'tech', 'ADBE': 'tech',
    'CRM': 'tech', 'AMD': 'tech', 'INTC': 'tech', 'QCOM': 'tech', 'TXN': 'tech',
    'CSCO': 'tech', 'IBM': 'tech', 'AMAT': 'tech', 'MU': 'tech', 'LRCX': 'tech',
    # Finance
    'JPM': 'finance', 'BAC': 'finance', 'WFC': 'finance', 'GS': 'finance', 'MS': 'finance',
    'C': 'finance', 'BLK': 'finance', 'SCHW': 'finance', 'AXP': 'finance', 'USB': 'finance',
    # Healthcare
    'UNH': 'healthcare', 'JNJ': 'healthcare', 'PFE': 'healthcare', 'MRK': 'healthcare',
    'ABBV': 'healthcare', 'LLY': 'healthcare', 'TMO': 'healthcare', 'ABT': 'healthcare',
    'DHR': 'healthcare', 'BMY': 'healthcare',
    # Consumer
    'WMT': 'consumer', 'PG': 'consumer', 'KO': 'consumer', 'PEP': 'consumer',
    'COST': 'consumer', 'MCD': 'consumer', 'NKE': 'consumer', 'SBUX': 'consumer',
    'HD': 'consumer', 'LOW': 'consumer',
    # Industrial
    'CAT': 'industrial', 'BA': 'industrial', 'GE': 'industrial', 'HON': 'industrial',
    'UPS': 'industrial', 'RTX': 'industrial', 'LMT': 'industrial', 'DE': 'industrial',
    'MMM': 'industrial', 'FDX': 'industrial',
    # Energy
    'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy', 'EOG': 'energy',
    'MPC': 'energy', 'PSX': 'energy', 'VLO': 'energy', 'OXY': 'energy', 'KMI': 'energy',
    # Communication
    'DIS': 'communication', 'CMCSA': 'communication', 'NFLX': 'communication',
    'T': 'communication', 'VZ': 'communication', 'TMUS': 'communication',
    'CHTR': 'communication', 'EA': 'communication', 'WBD': 'communication', 'PARA': 'communication',
    # Materials
    'LIN': 'materials', 'APD': 'materials', 'SHW': 'materials', 'ECL': 'materials',
    'FCX': 'materials', 'NEM': 'materials', 'NUE': 'materials', 'DOW': 'materials',
    'DD': 'materials', 'PPG': 'materials',
    # REITs & Utilities
    'AMT': 'reits', 'PLD': 'reits', 'CCI': 'reits', 'EQIX': 'reits', 'SPG': 'reits',
    'NEE': 'utilities', 'DUK': 'utilities', 'SO': 'utilities', 'D': 'utilities', 'AEP': 'utilities',
}

# Sectors where shorts work well (based on baseline analysis)
SHORT_FAVORABLE_SECTORS = {'energy', 'materials', 'healthcare', 'reits', 'utilities'}


def targeted_backtest(indicators: Dict, ticker: str,
                      base_threshold: float, ext_lb: int, hurst_lb: int, max_hurst: float,
                      min_exit: int, max_exit: int, inertia_sens: float,
                      duration_weight: float = 0.05, amplitude_weight: float = 0.5,
                      # Short parameters
                      short_threshold_mult: float = 1.0,
                      short_max_hurst_adj: float = 0.0,
                      short_exit_mult: float = 1.0,
                      short_min_duration: int = 0,
                      short_min_amp_ratio: float = 1.0,
                      target_vol: float = 0.15,
                      # Targeted short filters
                      short_min_beta: float = 0.8,
                      short_max_vol: float = 0.6,
                      short_min_zscore: float = 3.0,
                      ) -> Dict:
    """
    Targeted backtest - shorts only in favorable sectors with strict filters
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

    sector = SECTOR_MAP.get(ticker, 'unknown')
    can_short = sector in SHORT_FAVORABLE_SECTORS

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

        # Long signal - all sectors
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
                'sector': sector,
            })

        # Short signal - only favorable sectors with strict filters
        elif can_short and h < short_max_hurst and z > base_threshold_adj * short_threshold_mult:
            amp_ratio = amp / wave_amp_mean if wave_amp_mean > 0 else 1.0

            # Strict inertia requirements
            if dur < short_min_duration or amp_ratio < short_min_amp_ratio:
                continue

            # Beta filter - prefer high beta for shorts in these sectors
            if b < short_min_beta:
                continue

            # Volatility cap - avoid extremely volatile shorts
            if vol > short_max_vol:
                continue

            # Additional z-score filter for high confidence
            if z < short_min_zscore:
                continue

            short_exit_bars = int(np.clip(base_exit_bars * short_exit_mult, min_exit, max_exit))
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
                'sector': sector,
            })

    return {'trades': trades}


def test_config(args):
    """Test a single configuration"""
    precomputed, params = args
    all_trades = []

    for ticker, indicators in precomputed.items():
        result = targeted_backtest(indicators, ticker, **params)
        all_trades.extend(result['trades'])

    if len(all_trades) < 100:
        return None, None, 0

    result = aggregate_market_neutral_results(all_trades)

    if result['long_trades'] < 30 or result['short_trades'] < 20:  # Relaxed for targeted
        return None, None, 0

    trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])
    beta_penalty = abs(result['net_beta_exposure']) / result['total'] if result['total'] > 0 else 1

    min_hr = min(result['long_hit_rate'], result['short_hit_rate'])
    avg_hr = (result['long_hit_rate'] + result['short_hit_rate']) / 2

    # Score heavily weights minimum hit rate
    score = (min_hr * 0.8 + avg_hr * 0.2) * (0.3 + 0.7 * trade_ratio) * (1 - 0.05 * beta_penalty)

    return result, params, score


def main():
    """Targeted short optimization"""
    print("=" * 60)
    print("TARGETED SHORT OPTIMIZATION")
    print("Focus: Only short in favorable sectors")
    print("Sectors: Energy, Materials, Healthcare, REITs, Utilities")
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

    # Count favorable sector tickers
    favorable_count = sum(1 for t in precomputed.keys() if SECTOR_MAP.get(t, '') in SHORT_FAVORABLE_SECTORS)
    print(f"Favorable sector tickers for shorts: {favorable_count}")

    # Parameter grid - focused on what works
    print("\nBuilding targeted parameter grid...")
    configs = []

    for base_thresh in [3.0, 3.2, 3.5, 3.8, 4.0]:
        for max_hurst in [0.35, 0.38, 0.40, 0.42, 0.45]:
            for min_exit, max_exit in [(3, 5), (3, 6), (4, 6), (4, 7), (5, 7)]:
                # Short parameters - stricter for targeted approach
                for short_tm in [0.9, 1.0, 1.1, 1.2]:
                    for short_ha in [-0.05, -0.02, 0.0, 0.02]:
                        for short_em in [0.6, 0.8, 1.0]:
                            for short_md in [4, 5, 6]:
                                for short_mar in [1.3, 1.5, 1.7]:
                                    # Targeted filters
                                    for short_min_beta in [0.6, 0.8, 1.0]:
                                        for short_min_z in [3.0, 3.5, 4.0]:
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
                                                'short_max_vol': 0.6,
                                                'short_min_zscore': short_min_z,
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
        print("BEST TARGETED RESULT")
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
        for k in ['base_threshold', 'max_hurst', 'short_threshold_mult', 'short_max_hurst_adj',
                  'short_exit_mult', 'short_min_duration', 'short_min_amp_ratio',
                  'short_min_beta', 'short_min_zscore']:
            print(f"  {k}: {best_params[k]}")

    return {'result': best, 'params': best_params}


if __name__ == "__main__":
    result = main()
