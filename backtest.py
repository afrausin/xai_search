"""
Backtesting Framework for Extreme Hurst Signal
Pure Hurst with Dynamic Inertia - No RSI
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm
import warnings
import time
from scipy import stats
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS
from hurst_signal import (ExtremeHurstSignal, calculate_hurst, calculate_zscore,
                          calculate_atr, calculate_formation_duration, calculate_wave_amplitude)


def precompute_indicators(df: pd.DataFrame) -> Dict:
    """Precompute all indicators once per ticker"""
    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float) if 'High' in df.columns else close
    low = df['Low'].values.astype(float) if 'Low' in df.columns else close
    n = len(close)

    # Precompute z-scores at different lookbacks
    zscores = {}
    for lb in [10, 15, 20, 25]:
        zscores[lb] = calculate_zscore(close, lb)

    # Precompute Hurst at different lookbacks
    hursts = {}
    for hurst_lb in [15, 20, 25, 30]:
        hursts[hurst_lb] = calculate_hurst(close, hurst_lb)

    # ATR for volatility
    atr = calculate_atr(high, low, close)
    atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)

    # Formation duration and wave amplitude for inertia-based exits
    durations = {}
    for lb in [10, 15, 20]:
        z = zscores.get(lb, zscores[15])
        durations[lb] = calculate_formation_duration(close, z, threshold=1.0)

    wave_amp = calculate_wave_amplitude(close, lookback=20)

    return {
        'close': close,
        'n': n,
        'zscores': zscores,
        'hursts': hursts,
        'atr_ratio': atr_ratio,
        'durations': durations,
        'wave_amplitude': wave_amp,
    }


def fast_backtest_dynamic(indicators: Dict, base_threshold: float, ext_lb: int,
                          hurst_lb: int, max_hurst: float, min_exit: int, max_exit: int,
                          inertia_sens: float, duration_weight: float = 0.1,
                          amplitude_weight: float = 0.5) -> Dict:
    """
    Fast backtest with inertia-based dynamic entry and exit

    Inertia concept:
    - Formation duration: How long price has been extended (accumulated energy)
    - Wave amplitude: Magnitude of recent price swings (momentum)
    - Both affect the exit timing based on physical market intuition
    """
    close = indicators['close']
    n = indicators['n']
    zscore = indicators['zscores'].get(ext_lb, indicators['zscores'][15])
    hurst = indicators['hursts'].get(hurst_lb, indicators['hursts'][20])
    atr_ratio = indicators['atr_ratio']
    duration = indicators['durations'].get(ext_lb, indicators['durations'][15])
    wave_amp = indicators['wave_amplitude']

    long_wins = []
    short_wins = []

    min_idx = max(hurst_lb, ext_lb, 25)
    base_bars = (min_exit + max_exit) / 2

    # Normalize wave amplitude
    wave_amp_mean = np.mean(wave_amp[wave_amp > 0]) if np.any(wave_amp > 0) else 1.0

    for i in range(min_idx, n - max_exit):
        h = hurst[i]
        z = zscore[i]
        vol_adj = atr_ratio[i]
        dur = duration[i]
        amp = wave_amp[i]

        if h >= max_hurst:
            continue

        # Dynamic threshold based on Hurst and volatility
        hurst_factor = 0.5 + h * inertia_sens
        threshold = base_threshold * hurst_factor * (0.8 + vol_adj * 0.4)

        # Inertia-based exit calculation:
        # 1. Hurst factor: Lower Hurst = faster mean reversion = shorter hold
        # 2. Duration factor: Longer formations have more energy = may take longer to unwind
        # 3. Amplitude factor: Larger waves = more momentum to dissipate
        duration_factor = 1.0 + dur * duration_weight  # More bars in formation = longer exit
        amplitude_factor = 1.0 + (amp / wave_amp_mean - 1.0) * amplitude_weight if wave_amp_mean > 0 else 1.0

        # Combined inertia calculation
        # Lower Hurst speeds up exit, longer duration/higher amplitude slow it down
        exit_bars = int(np.clip(
            base_bars * hurst_factor * duration_factor * amplitude_factor,
            min_exit, max_exit
        ))

        if i + exit_bars >= n:
            continue

        # Long signal
        if z < -threshold:
            entry = close[i]
            exit_price = close[i + exit_bars]
            ret = (exit_price - entry) / entry
            long_wins.append(ret > 0)

        # Short signal
        elif z > threshold:
            entry = close[i]
            exit_price = close[i + exit_bars]
            ret = (entry - exit_price) / entry
            short_wins.append(ret > 0)

    return {
        'long_trades': len(long_wins),
        'short_trades': len(short_wins),
        'long_wins': sum(long_wins),
        'short_wins': sum(short_wins),
    }


def run_fast_backtest(precomputed: Dict[str, Dict], **params) -> Dict:
    """Run backtest across all tickers"""
    total_long = 0
    total_short = 0
    long_wins = 0
    short_wins = 0

    for ticker, indicators in precomputed.items():
        result = fast_backtest_dynamic(indicators, **params)
        total_long += result['long_trades']
        total_short += result['short_trades']
        long_wins += result['long_wins']
        short_wins += result['short_wins']

    if total_long < 5 or total_short < 5:
        return {'long_hit_rate': 0, 'short_hit_rate': 0, 'total': 0}

    return {
        'long_trades': total_long,
        'short_trades': total_short,
        'long_hit_rate': (long_wins / total_long * 100) if total_long > 0 else 0,
        'short_hit_rate': (short_wins / total_short * 100) if total_short > 0 else 0,
        'total': total_long + total_short,
    }


def test_single_config(args):
    """Test a single configuration (for parallel processing)"""
    precomputed, params = args
    result = run_fast_backtest(precomputed, **params)
    # Require at least 100 trades for statistical significance
    if result['total'] < 100:
        return None, None, 0
    # Also require at least 30 trades each side
    if result.get('long_trades', 0) < 30 or result.get('short_trades', 0) < 30:
        return None, None, 0
    min_hr = min(result['long_hit_rate'], result['short_hit_rate'])
    return result, params, min_hr


def optimize_signal(precomputed: Dict[str, Dict], target: float = 60.0) -> Dict:
    """Optimize pure Hurst signal with dynamic inertia using parallel processing"""
    from multiprocessing import Pool, cpu_count
    import itertools

    print(f"\n{'='*60}")
    print(f"PURE HURST WITH DYNAMIC INERTIA")
    print(f"Target: {target}% hit rate on both longs and shorts")
    print(f"{'='*60}\n")

    # Balanced search - focusing on min(long_hr, short_hr) with sufficient trades
    base_thresholds = [3.0, 3.2, 3.5, 3.8, 4.0]
    ext_lbs = [10, 15]
    hurst_lbs = [15, 20]
    max_hursts = [0.38, 0.40, 0.42, 0.45]  # Slightly relaxed for more trades
    min_exits = [3, 4, 5]
    max_exits = [5, 6, 8]
    inertia_sensitivities = [0.6, 0.8, 1.0]
    duration_weights = [0.02, 0.05, 0.08]
    amplitude_weights = [0.2, 0.35, 0.5]

    # Generate all valid parameter combinations
    all_configs = []
    for base_thresh, ext_lb, hurst_lb, max_hurst, min_exit, max_exit, inertia, dur_w, amp_w in itertools.product(
        base_thresholds, ext_lbs, hurst_lbs, max_hursts, min_exits, max_exits,
        inertia_sensitivities, duration_weights, amplitude_weights
    ):
        if min_exit >= max_exit:
            continue
        params = {
            'base_threshold': base_thresh,
            'ext_lb': ext_lb,
            'hurst_lb': hurst_lb,
            'max_hurst': max_hurst,
            'min_exit': min_exit,
            'max_exit': max_exit,
            'inertia_sens': inertia,
            'duration_weight': dur_w,
            'amplitude_weight': amp_w,
        }
        all_configs.append((precomputed, params))

    print(f"Testing {len(all_configs)} configurations with {cpu_count()} cores...")

    best = None
    best_params = None
    best_min = 0
    found_target = False

    # Process in batches to check for early stopping
    batch_size = 500
    for batch_start in tqdm(range(0, len(all_configs), batch_size), desc="Batches"):
        batch = all_configs[batch_start:batch_start + batch_size]

        # Run batch in parallel
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(test_single_config, batch)

        for result, params, min_hr in results:
            if result is None:
                continue

            if min_hr > best_min:
                best_min = min_hr
                best = result
                best_params = params.copy()
                print(f"\n  New best: Long {result['long_hit_rate']:.1f}%, Short {result['short_hit_rate']:.1f}% (Trades: {result['total']})")

                if min_hr >= target:
                    print(f"\n✓ TARGET ACHIEVED!")
                    print(f"  Long: {result['long_hit_rate']:.1f}%")
                    print(f"  Short: {result['short_hit_rate']:.1f}%")
                    print(f"  Trades: {result['total']}")
                    found_target = True
                    break

        if found_target:
            break

    if best and not found_target:
        print(f"\nBest achieved:")
        print(f"  Long: {best['long_hit_rate']:.1f}%, Short: {best['short_hit_rate']:.1f}%")
        print(f"  Trades: {best['total']}")

    return {'result': best, 'params': best_params}


def main():
    """Main execution"""
    print("=" * 60)
    print("EXTREME HURST TRADING SIGNAL")
    print("Pure Hurst with Dynamic Inertia - No RSI")
    print("=" * 60)

    # Download data
    data = download_all_data(TICKERS, period="2y")

    if len(data) < 50:
        print("Retrying data download...")
        time.sleep(3)
        data = download_all_data(TICKERS, period="2y")

    print(f"\nDownloaded {len(data)} tickers")

    # Precompute all indicators
    print("\nPrecomputing indicators...")
    precomputed = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            precomputed[ticker] = precompute_indicators(df)
        except:
            continue

    print(f"Precomputed indicators for {len(precomputed)} tickers")

    # Run optimization
    result = optimize_signal(precomputed, target=60.0)

    if result['result']:
        r = result['result']
        p = result['params']

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Total Trades: {r['total']}")
        print(f"Long Trades: {r['long_trades']}")
        print(f"Short Trades: {r['short_trades']}")
        print(f"Long Hit Rate: {r['long_hit_rate']:.2f}%")
        print(f"Short Hit Rate: {r['short_hit_rate']:.2f}%")

        print(f"\nOptimal Parameters (Dynamic Inertia):")
        print(f"  Base Threshold: {p['base_threshold']:.2f}")
        print(f"  Extension Lookback: {p['ext_lb']}")
        print(f"  Hurst Lookback: {p['hurst_lb']}")
        print(f"  Max Hurst: {p['max_hurst']:.2f}")
        print(f"  Min Exit Bars: {p['min_exit']}")
        print(f"  Max Exit Bars: {p['max_exit']}")
        print(f"  Inertia Sensitivity: {p['inertia_sens']:.2f}")

    return result


if __name__ == "__main__":
    result = main()
