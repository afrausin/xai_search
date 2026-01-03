"""
Regime-Based Short Signal Optimization
Focus on finding market conditions where shorts work best
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS, fetch_yahoo_data
from hurst_signal import (calculate_hurst, calculate_zscore,
                          calculate_atr, calculate_formation_duration, calculate_wave_amplitude)


def compute_regime_indicators(close: np.ndarray, market_returns: np.ndarray) -> Dict:
    """Compute regime indicators for filtering"""
    n = len(close)

    # Stock returns
    stock_returns = np.zeros(n)
    stock_returns[1:] = (close[1:] - close[:-1]) / close[:-1]

    # Rolling market volatility (realized vol of last 20 days)
    market_vol = np.zeros(n)
    for i in range(20, n):
        market_vol[i] = np.std(market_returns[i-20:i]) * np.sqrt(252)

    # Market momentum (last 5 days cumulative return)
    market_momentum = np.zeros(n)
    for i in range(5, n):
        market_momentum[i] = np.sum(market_returns[i-5:i])

    # Stock relative strength vs market
    rel_strength = np.zeros(n)
    for i in range(20, n):
        stock_cum = np.sum(stock_returns[i-20:i])
        market_cum = np.sum(market_returns[i-20:i])
        rel_strength[i] = stock_cum - market_cum

    # Cross-sectional momentum (vs own recent trend)
    stock_momentum_5d = np.zeros(n)
    stock_momentum_20d = np.zeros(n)
    for i in range(20, n):
        stock_momentum_5d[i] = np.sum(stock_returns[i-5:i])
        stock_momentum_20d[i] = np.sum(stock_returns[i-20:i])

    # Mean reversion setup: short-term vs long-term trend divergence
    mean_rev_setup = stock_momentum_5d - stock_momentum_20d / 4  # 5d vs 5d-equivalent of 20d

    return {
        'market_vol': market_vol,
        'market_momentum': market_momentum,
        'rel_strength': rel_strength,
        'stock_momentum_5d': stock_momentum_5d,
        'mean_rev_setup': mean_rev_setup,
    }


def precompute_all_indicators(df: pd.DataFrame, market_returns: np.ndarray) -> Dict:
    """Precompute all indicators for regime-based backtest"""
    close = df['Close'].values.astype(float).flatten()
    high = df['High'].values.astype(float).flatten() if 'High' in df.columns else close
    low = df['Low'].values.astype(float).flatten() if 'Low' in df.columns else close
    n = len(close)

    # Core indicators
    zscores = {}
    for lb in [10, 15, 20]:
        zscores[lb] = calculate_zscore(close, lb)

    hursts = {}
    for hurst_lb in [15, 20, 25]:
        hursts[hurst_lb] = calculate_hurst(close, hurst_lb)

    atr = calculate_atr(high, low, close)
    atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)

    durations = {}
    for lb in [10, 15]:
        z = zscores.get(lb, zscores[15])
        durations[lb] = calculate_formation_duration(close, z, threshold=1.0)

    wave_amp = calculate_wave_amplitude(close, lookback=20)

    # Beta and volatility
    stock_returns = np.zeros(n)
    stock_returns[1:] = (close[1:] - close[:-1]) / close[:-1]

    beta = np.zeros(n)
    volatility = np.zeros(n)
    for i in range(60, n):
        sr = stock_returns[i-60:i]
        mr = market_returns[i-60:i]
        cov = np.cov(sr, mr)
        if cov.shape == (2, 2) and cov[1, 1] > 0:
            beta[i] = cov[0, 1] / cov[1, 1]
        volatility[i] = np.std(sr) * np.sqrt(252)

    # Regime indicators
    regime = compute_regime_indicators(close, market_returns)

    return {
        'close': close,
        'n': n,
        'zscores': zscores,
        'hursts': hursts,
        'atr_ratio': atr_ratio,
        'durations': durations,
        'wave_amplitude': wave_amp,
        'beta': beta,
        'volatility': volatility,
        'market_vol': regime['market_vol'],
        'market_momentum': regime['market_momentum'],
        'rel_strength': regime['rel_strength'],
        'stock_momentum_5d': regime['stock_momentum_5d'],
        'mean_rev_setup': regime['mean_rev_setup'],
    }


def regime_backtest(indicators: Dict, base_threshold: float, ext_lb: int,
                    hurst_lb: int, max_hurst: float, min_exit: int, max_exit: int,
                    inertia_sens: float, duration_weight: float = 0.05,
                    amplitude_weight: float = 0.5,
                    # Short-specific params
                    short_threshold_mult: float = 1.0,
                    short_min_duration: int = 3,
                    short_min_amp_ratio: float = 1.2,
                    # Regime filters for shorts
                    short_min_market_vol: float = 0.0,
                    short_max_market_vol: float = 1.0,
                    short_min_rel_strength: float = 0.0,
                    short_min_mean_rev_setup: float = 0.0,
                    # Exit params
                    short_exit_mult: float = 1.0,
                    target_vol: float = 0.15) -> Dict:
    """
    Regime-based backtest with market condition filters for shorts

    Key insight: Shorts work better when:
    1. Market volatility is in a specific range (not too low, not too high)
    2. Stock has outperformed market recently (relative strength)
    3. Short-term momentum exceeds long-term momentum (mean-rev setup)
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

    # Regime indicators
    market_vol = indicators['market_vol']
    rel_strength = indicators['rel_strength']
    mean_rev_setup = indicators['mean_rev_setup']

    trades = []

    min_idx = max(hurst_lb, ext_lb, 60, 25)
    base_bars = (min_exit + max_exit) / 2

    wave_amp_mean = np.mean(wave_amp[wave_amp > 0]) if np.any(wave_amp > 0) else 1.0

    for i in range(min_idx, n - max_exit):
        h = hurst[i]
        z = zscore[i]
        vol_adj = atr_ratio[i]
        dur = duration[i]
        amp = wave_amp[i]
        b = beta[i]
        vol = volatility[i]
        m_vol = market_vol[i]
        rel_str = rel_strength[i]
        mr_setup = mean_rev_setup[i]

        if np.isnan(vol) or vol <= 0 or np.isnan(b):
            continue

        position_size = min(target_vol / vol, 2.0)

        hurst_factor = 0.5 + h * inertia_sens
        base_threshold_adj = base_threshold * hurst_factor * (0.8 + vol_adj * 0.4)

        duration_factor = 1.0 + dur * duration_weight
        amplitude_factor = 1.0 + (amp / wave_amp_mean - 1.0) * amplitude_weight if wave_amp_mean > 0 else 1.0
        base_exit_bars = base_bars * hurst_factor * duration_factor * amplitude_factor

        # Long signal (keep simple)
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

        # Short signal with regime filters
        elif h < max_hurst and z > base_threshold_adj * short_threshold_mult:
            amp_ratio = amp / wave_amp_mean if wave_amp_mean > 0 else 1.0

            # Basic inertia filter
            if dur < short_min_duration or amp_ratio < short_min_amp_ratio:
                continue

            # Regime filters
            if m_vol < short_min_market_vol or m_vol > short_max_market_vol:
                continue

            if rel_str < short_min_rel_strength:
                continue

            if mr_setup < short_min_mean_rev_setup:
                continue

            # Calculate exit
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
            })

    return {'trades': trades}


def aggregate_results(trades: List[Dict]) -> Dict:
    """Aggregate trade results"""
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']

    return {
        'total': len(trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_wins': sum(1 for t in long_trades if t['win']),
        'short_wins': sum(1 for t in short_trades if t['win']),
        'long_hit_rate': sum(1 for t in long_trades if t['win']) / len(long_trades) * 100 if long_trades else 0,
        'short_hit_rate': sum(1 for t in short_trades if t['win']) / len(short_trades) * 100 if short_trades else 0,
        'long_pnl': sum(t['return'] * t['position_size'] for t in long_trades),
        'short_pnl': sum(t['return'] * t['position_size'] for t in short_trades),
        'net_beta_exposure': sum(t['beta_exposure'] for t in trades),
    }


def test_config(args):
    """Test a single configuration"""
    precomputed, params = args
    all_trades = []

    for ticker, indicators in precomputed.items():
        result = regime_backtest(indicators, **params)
        all_trades.extend(result['trades'])

    if len(all_trades) < 100:
        return None, None, 0, 0

    result = aggregate_results(all_trades)

    if result['long_trades'] < 50 or result['short_trades'] < 50:
        return None, None, 0, 0

    trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])

    # Require reasonable balance
    if trade_ratio < 0.3:
        return None, None, 0, 0

    min_hr = min(result['long_hit_rate'], result['short_hit_rate'])

    # Score: min_hr weighted by balance
    score = min_hr * (0.5 + 0.5 * trade_ratio)

    return result, params, score, min_hr


def main():
    """Regime-based optimization"""
    print("=" * 60)
    print("REGIME-BASED SHORT OPTIMIZATION")
    print("Focus: Find market conditions where shorts work better")
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
    print("\nPrecomputing regime indicators...")
    precomputed = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            precomputed[ticker] = precompute_all_indicators(df, market_returns)
        except:
            continue
    print(f"Precomputed {len(precomputed)} tickers")

    # Parameter grid
    print("\nBuilding regime-based parameter grid...")
    configs = []

    for base_thresh in [3.0, 3.2, 3.5]:
        for max_hurst in [0.38, 0.40, 0.42]:
            for min_exit, max_exit in [(3, 6), (4, 7)]:
                for short_tm in [1.0, 1.05, 1.1]:
                    for short_md in [2, 3, 4]:
                        for short_mar in [1.0, 1.1, 1.2]:
                            # Regime filters - focus on finding the right ranges
                            for min_mv in [0.0, 0.10, 0.15]:  # Min market vol
                                for max_mv in [0.25, 0.30, 0.40, 1.0]:  # Max market vol
                                    for min_rs in [0.0, 0.02, 0.04]:  # Min relative strength
                                        for min_mrs in [0.0, 0.01, 0.02]:  # Min mean-rev setup
                                            if min_mv >= max_mv:
                                                continue
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
                                                'short_min_duration': short_md,
                                                'short_min_amp_ratio': short_mar,
                                                'short_min_market_vol': min_mv,
                                                'short_max_market_vol': max_mv,
                                                'short_min_rel_strength': min_rs,
                                                'short_min_mean_rev_setup': min_mrs,
                                                'short_exit_mult': 0.8,
                                                'target_vol': 0.15,
                                            }
                                            configs.append((precomputed, params))

    print(f"Testing {len(configs)} configurations with {cpu_count()} cores...")

    # Run optimization
    best = None
    best_params = None
    best_score = 0
    best_min_hr = 0

    batch_size = 200
    for batch_start in tqdm(range(0, len(configs), batch_size), desc="Batches"):
        batch = configs[batch_start:batch_start + batch_size]

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(test_config, batch)

        for result, params, score, min_hr in results:
            if result is None:
                continue

            if min_hr > best_min_hr or (min_hr == best_min_hr and score > best_score):
                best_min_hr = min_hr
                best_score = score
                best = result
                best_params = params.copy()
                trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])
                print(f"\n  New best: L:{result['long_hit_rate']:.1f}% S:{result['short_hit_rate']:.1f}% "
                      f"Min:{min_hr:.1f}% ({result['long_trades']}L/{result['short_trades']}S) "
                      f"Bal:{trade_ratio:.2f}")

    if best:
        trade_ratio = min(best['long_trades'], best['short_trades']) / max(best['long_trades'], best['short_trades'])
        min_hr = min(best['long_hit_rate'], best['short_hit_rate'])

        print("\n" + "=" * 60)
        print("BEST REGIME-BASED RESULT")
        print("=" * 60)
        print(f"Total Trades: {best['total']}")
        print(f"  Long: {best['long_trades']}, Short: {best['short_trades']}")
        print(f"  Trade Balance: {trade_ratio:.2f}")
        print(f"\nHit Rates:")
        print(f"  Long:    {best['long_hit_rate']:.2f}%")
        print(f"  Short:   {best['short_hit_rate']:.2f}%")
        print(f"  Minimum: {min_hr:.2f}%")
        print(f"\nKey Regime Parameters:")
        print(f"  Market Vol Range: [{best_params['short_min_market_vol']:.2f}, {best_params['short_max_market_vol']:.2f}]")
        print(f"  Min Relative Strength: {best_params['short_min_rel_strength']:.2f}")
        print(f"  Min Mean-Rev Setup: {best_params['short_min_mean_rev_setup']:.2f}")

    return {'result': best, 'params': best_params}


if __name__ == "__main__":
    result = main()
