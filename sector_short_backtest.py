"""
Sector-Based Short Optimization
Hypothesis: Shorts may work better in certain sectors (high-beta tech/growth)
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


# Define sectors for each ticker
SECTOR_MAP = {
    # Mega-cap Tech - typically higher beta, good for shorts
    'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech', 'NVDA': 'tech',
    'META': 'tech', 'TSLA': 'tech', 'AVGO': 'tech', 'ORCL': 'tech', 'ADBE': 'tech',
    'CRM': 'tech', 'AMD': 'tech', 'INTC': 'tech', 'QCOM': 'tech', 'TXN': 'tech',
    'CSCO': 'tech', 'IBM': 'tech', 'AMAT': 'tech', 'MU': 'tech', 'LRCX': 'tech',
    # Finance - cyclical
    'JPM': 'finance', 'BAC': 'finance', 'WFC': 'finance', 'GS': 'finance', 'MS': 'finance',
    'C': 'finance', 'BLK': 'finance', 'SCHW': 'finance', 'AXP': 'finance', 'USB': 'finance',
    # Healthcare - defensive, less mean-reverting
    'UNH': 'healthcare', 'JNJ': 'healthcare', 'PFE': 'healthcare', 'MRK': 'healthcare',
    'ABBV': 'healthcare', 'LLY': 'healthcare', 'TMO': 'healthcare', 'ABT': 'healthcare',
    'DHR': 'healthcare', 'BMY': 'healthcare',
    # Consumer - mixed
    'WMT': 'consumer', 'PG': 'consumer', 'KO': 'consumer', 'PEP': 'consumer', 'COST': 'consumer',
    'MCD': 'consumer', 'NKE': 'consumer', 'SBUX': 'consumer', 'HD': 'consumer', 'LOW': 'consumer',
    # Industrial - cyclical
    'CAT': 'industrial', 'BA': 'industrial', 'GE': 'industrial', 'HON': 'industrial',
    'UPS': 'industrial', 'RTX': 'industrial', 'LMT': 'industrial', 'DE': 'industrial',
    'MMM': 'industrial', 'FDX': 'industrial',
    # Energy - high volatility
    'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy', 'EOG': 'energy',
    'MPC': 'energy', 'PSX': 'energy', 'VLO': 'energy', 'OXY': 'energy', 'KMI': 'energy',
    # Communication - mixed
    'DIS': 'communication', 'CMCSA': 'communication', 'NFLX': 'communication', 'T': 'communication',
    'VZ': 'communication', 'TMUS': 'communication', 'CHTR': 'communication', 'EA': 'communication',
    'WBD': 'communication', 'PARA': 'communication',
    # Materials
    'LIN': 'materials', 'APD': 'materials', 'SHW': 'materials', 'ECL': 'materials',
    'FCX': 'materials', 'NEM': 'materials', 'NUE': 'materials', 'DOW': 'materials',
    'DD': 'materials', 'PPG': 'materials',
    # Real Estate & Utilities - low volatility, poor for shorts
    'AMT': 'reits', 'PLD': 'reits', 'CCI': 'reits', 'EQIX': 'reits', 'SPG': 'reits',
    'NEE': 'utilities', 'DUK': 'utilities', 'SO': 'utilities', 'D': 'utilities', 'AEP': 'utilities',
}

# Sectors where shorts might work better (high beta, more volatile)
HIGH_BETA_SECTORS = {'tech', 'energy', 'finance', 'materials', 'communication'}
# Sectors where shorts might struggle (defensive)
DEFENSIVE_SECTORS = {'healthcare', 'consumer', 'utilities', 'reits'}


def sector_short_backtest(indicators: Dict, ticker: str, base_threshold: float, ext_lb: int,
                          hurst_lb: int, max_hurst: float, min_exit: int, max_exit: int,
                          inertia_sens: float, duration_weight: float = 0.1,
                          amplitude_weight: float = 0.5,
                          short_threshold_mult: float = 1.0,
                          short_max_hurst_adj: float = 0.0,
                          short_exit_mult: float = 1.0,
                          short_min_duration: int = 0,
                          short_min_amp_ratio: float = 1.0,
                          target_vol: float = 0.15,
                          short_only_high_beta: bool = False,
                          short_exclude_defensive: bool = False) -> Dict:
    """
    Sector-aware backtest with optional sector filtering for shorts
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

    # Get sector for this ticker
    sector = SECTOR_MAP.get(ticker, 'unknown')
    is_high_beta_sector = sector in HIGH_BETA_SECTORS
    is_defensive_sector = sector in DEFENSIVE_SECTORS

    # Decide if shorts are allowed for this ticker
    allow_shorts = True
    if short_only_high_beta and not is_high_beta_sector:
        allow_shorts = False
    if short_exclude_defensive and is_defensive_sector:
        allow_shorts = False

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

        # Long signal - all sectors allowed
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

        # Short signal - sector filtered
        elif allow_shorts and h < short_max_hurst and z > base_threshold_adj * short_threshold_mult:
            amp_ratio = amp / wave_amp_mean if wave_amp_mean > 0 else 1.0

            if dur < short_min_duration or amp_ratio < short_min_amp_ratio:
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
        result = sector_short_backtest(indicators, ticker, **params)
        all_trades.extend(result['trades'])

    if len(all_trades) < 100:
        return None, None, 0

    result = aggregate_market_neutral_results(all_trades)

    if result['long_trades'] < 30 or result['short_trades'] < 30:
        return None, None, 0

    trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])
    beta_penalty = abs(result['net_beta_exposure']) / result['total'] if result['total'] > 0 else 1

    # Score: prioritize MINIMUM hit rate with balance
    min_hr = min(result['long_hit_rate'], result['short_hit_rate'])
    avg_hr = (result['long_hit_rate'] + result['short_hit_rate']) / 2

    # Require minimum balance for valid result
    if trade_ratio < 0.2:
        return None, None, 0

    score = (min_hr * 0.7 + avg_hr * 0.3) * (0.5 + 0.5 * trade_ratio) * (1 - 0.05 * beta_penalty)

    return result, params, score


def main():
    """Sector-based short optimization"""
    print("=" * 60)
    print("SECTOR-BASED SHORT OPTIMIZATION")
    print("Focus: Find sectors where shorts work better")
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

    # First analyze sector performance
    print("\n" + "=" * 60)
    print("SECTOR ANALYSIS - Baseline")
    print("=" * 60)

    baseline_params = {
        'base_threshold': 3.5,
        'ext_lb': 15,
        'hurst_lb': 20,
        'max_hurst': 0.40,
        'min_exit': 4,
        'max_exit': 7,
        'inertia_sens': 0.8,
        'duration_weight': 0.05,
        'amplitude_weight': 0.5,
        'short_threshold_mult': 1.0,
        'short_max_hurst_adj': 0.0,
        'short_exit_mult': 1.0,
        'short_min_duration': 3,
        'short_min_amp_ratio': 1.2,
        'target_vol': 0.15,
        'short_only_high_beta': False,
        'short_exclude_defensive': False,
    }

    # Collect all trades by sector
    all_trades = []
    for ticker, indicators in precomputed.items():
        result = sector_short_backtest(indicators, ticker, **baseline_params)
        all_trades.extend(result['trades'])

    # Analyze by sector
    sector_stats = {}
    for trade in all_trades:
        sector = trade['sector']
        direction = trade['direction']
        key = f"{sector}_{direction}"

        if key not in sector_stats:
            sector_stats[key] = {'wins': 0, 'total': 0}

        sector_stats[key]['total'] += 1
        if trade['win']:
            sector_stats[key]['wins'] += 1

    print("\nSector Performance:")
    print(f"{'Sector':<15} {'Long Trades':<12} {'Long HR':<10} {'Short Trades':<13} {'Short HR':<10}")
    print("-" * 60)

    for sector in sorted(set(t['sector'] for t in all_trades)):
        long_key = f"{sector}_long"
        short_key = f"{sector}_short"

        long_stats = sector_stats.get(long_key, {'wins': 0, 'total': 0})
        short_stats = sector_stats.get(short_key, {'wins': 0, 'total': 0})

        long_hr = (long_stats['wins'] / long_stats['total'] * 100) if long_stats['total'] > 0 else 0
        short_hr = (short_stats['wins'] / short_stats['total'] * 100) if short_stats['total'] > 0 else 0

        print(f"{sector:<15} {long_stats['total']:<12} {long_hr:<10.1f} {short_stats['total']:<13} {short_hr:<10.1f}")

    # Now optimize with sector filtering
    print("\n" + "=" * 60)
    print("OPTIMIZING WITH SECTOR FILTERS")
    print("=" * 60)

    configs = []

    # Test different sector filtering strategies
    sector_filters = [
        (False, False),   # No filter
        (True, False),    # Only high-beta sectors for shorts
        (False, True),    # Exclude defensive sectors for shorts
        (True, True),     # Both
    ]

    for base_thresh in [3.0, 3.2, 3.5, 3.8]:
        for max_hurst in [0.35, 0.40, 0.45]:
            for min_exit, max_exit in [(3, 6), (4, 7), (5, 8)]:
                for short_tm in [1.0, 1.1, 1.2]:
                    for short_ha in [-0.02, 0.0, 0.02]:
                        for short_em in [0.7, 0.9, 1.0]:
                            for short_md in [3, 4, 5]:
                                for short_mar in [1.2, 1.4, 1.6]:
                                    for only_hb, excl_def in sector_filters:
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
                                            'short_only_high_beta': only_hb,
                                            'short_exclude_defensive': excl_def,
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
                filter_str = ""
                if params['short_only_high_beta']:
                    filter_str += " HB"
                if params['short_exclude_defensive']:
                    filter_str += " !DEF"
                print(f"\n  New best: L:{result['long_hit_rate']:.1f}% S:{result['short_hit_rate']:.1f}% "
                      f"Min:{min_hr:.1f}% ({result['long_trades']}L/{result['short_trades']}S) "
                      f"Bal:{trade_ratio:.2f}{filter_str}")

    if best:
        trade_ratio = min(best['long_trades'], best['short_trades']) / max(best['long_trades'], best['short_trades'])
        min_hr = min(best['long_hit_rate'], best['short_hit_rate'])

        print("\n" + "=" * 60)
        print("BEST SECTOR-FILTERED RESULT")
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
        print(f"\nSector Filters:")
        print(f"  Short only high-beta: {best_params['short_only_high_beta']}")
        print(f"  Exclude defensive: {best_params['short_exclude_defensive']}")
        print(f"\nKey Parameters:")
        for k in ['short_threshold_mult', 'short_max_hurst_adj', 'short_exit_mult',
                  'short_min_duration', 'short_min_amp_ratio']:
            print(f"  {k}: {best_params[k]}")

    return {'result': best, 'params': best_params}


if __name__ == "__main__":
    result = main()
