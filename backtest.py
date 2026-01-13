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
                          amplitude_weight: float = 0.5,
                          short_threshold_mult: float = 1.0,
                          short_max_hurst_adj: float = 0.0,
                          short_exit_mult: float = 1.0,
                          short_min_duration: int = 0,
                          short_min_amp_ratio: float = 1.0) -> Dict:
    """
    Fast backtest with inertia-based dynamic entry and exit

    Inertia concept:
    - Formation duration: How long price has been extended (accumulated energy)
    - Wave amplitude: Magnitude of recent price swings (momentum)
    - Both affect the exit timing based on physical market intuition

    Asymmetric parameters for long/short:
    - short_threshold_mult: Multiply base threshold for shorts (>1 = stricter)
    - short_max_hurst_adj: Lower the max_hurst for shorts (more negative = stricter)
    - short_exit_mult: Multiply exit bars for shorts (different hold times)
    - short_min_duration: Minimum bars in extended state before shorting
    - short_min_amp_ratio: Minimum wave amplitude ratio (vs mean) for shorts
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

    # Asymmetric parameters
    short_max_hurst = max_hurst + short_max_hurst_adj

    # Normalize wave amplitude
    wave_amp_mean = np.mean(wave_amp[wave_amp > 0]) if np.any(wave_amp > 0) else 1.0

    for i in range(min_idx, n - max_exit):
        h = hurst[i]
        z = zscore[i]
        vol_adj = atr_ratio[i]
        dur = duration[i]
        amp = wave_amp[i]

        # Dynamic threshold based on Hurst and volatility
        hurst_factor = 0.5 + h * inertia_sens
        base_threshold_adj = base_threshold * hurst_factor * (0.8 + vol_adj * 0.4)

        # Inertia-based exit calculation:
        # 1. Hurst factor: Lower Hurst = faster mean reversion = shorter hold
        # 2. Duration factor: Longer formations have more energy = may take longer to unwind
        # 3. Amplitude factor: Larger waves = more momentum to dissipate
        duration_factor = 1.0 + dur * duration_weight  # More bars in formation = longer exit
        amplitude_factor = 1.0 + (amp / wave_amp_mean - 1.0) * amplitude_weight if wave_amp_mean > 0 else 1.0

        # Combined inertia calculation
        # Lower Hurst speeds up exit, longer duration/higher amplitude slow it down
        base_exit_bars = base_bars * hurst_factor * duration_factor * amplitude_factor

        # Long signal
        if h < max_hurst and z < -base_threshold_adj:
            exit_bars = int(np.clip(base_exit_bars, min_exit, max_exit))
            if i + exit_bars >= n:
                continue
            entry = close[i]
            exit_price = close[i + exit_bars]
            ret = (exit_price - entry) / entry
            long_wins.append(ret > 0)

        # Short signal - with asymmetric adjustments and inertia requirements
        elif h < short_max_hurst and z > base_threshold_adj * short_threshold_mult:
            # Require minimum accumulated inertia for shorts
            amp_ratio = amp / wave_amp_mean if wave_amp_mean > 0 else 1.0
            if dur < short_min_duration or amp_ratio < short_min_amp_ratio:
                continue
            short_exit_bars = int(np.clip(base_exit_bars * short_exit_mult, min_exit, max_exit))
            if i + short_exit_bars >= n:
                continue
            entry = close[i]
            exit_price = close[i + short_exit_bars]
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
    print(f"PURE HURST WITH DYNAMIC INERTIA (ASYMMETRIC)")
    print(f"Target: {target}% hit rate on both longs and shorts")
    print(f"{'='*60}\n")

    # Core parameters - refined search near optimum
    base_thresholds = [3.2, 3.5, 3.8]
    ext_lbs = [15]
    hurst_lbs = [20]
    max_hursts = [0.35, 0.38, 0.40]
    min_exits = [3, 4, 5]
    max_exits = [6, 7, 8]
    inertia_sensitivities = [0.6, 0.8, 1.0]
    duration_weights = [0.02, 0.05]
    amplitude_weights = [0.35, 0.5]

    # Asymmetric parameters for shorts - focus on what works
    short_threshold_mults = [1.0, 1.1]  # Require higher z-score for shorts
    short_max_hurst_adjs = [0.0, -0.02, -0.05]  # Stricter Hurst filter for shorts
    short_exit_mults = [0.8, 1.0, 1.2]  # Different hold times for shorts

    # Inertia requirements for shorts (key innovation) - fine-tune around 5 duration, 1.5 amp
    short_min_durations = [3, 4, 5, 6]  # Minimum bars in extended state before shorting
    short_min_amp_ratios = [1.2, 1.3, 1.4, 1.5]  # Minimum amplitude ratio for shorts

    # Generate all valid parameter combinations
    all_configs = []
    for base_thresh, ext_lb, hurst_lb, max_hurst, min_exit, max_exit, inertia, dur_w, amp_w, short_tm, short_ha, short_em, short_md, short_mar in itertools.product(
        base_thresholds, ext_lbs, hurst_lbs, max_hursts, min_exits, max_exits,
        inertia_sensitivities, duration_weights, amplitude_weights,
        short_threshold_mults, short_max_hurst_adjs, short_exit_mults,
        short_min_durations, short_min_amp_ratios
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
            'short_threshold_mult': short_tm,
            'short_max_hurst_adj': short_ha,
            'short_exit_mult': short_em,
            'short_min_duration': short_md,
            'short_min_amp_ratio': short_mar,
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
        print(f"  Max Hurst (Long): {p['max_hurst']:.2f}")
        print(f"  Min Exit Bars: {p['min_exit']}")
        print(f"  Max Exit Bars: {p['max_exit']}")
        print(f"  Inertia Sensitivity: {p['inertia_sens']:.2f}")
        print(f"  Duration Weight: {p.get('duration_weight', 0.05):.2f}")
        print(f"  Amplitude Weight: {p.get('amplitude_weight', 0.35):.2f}")
        if 'short_threshold_mult' in p:
            print(f"\n  Asymmetric Short Parameters:")
            print(f"    Short Threshold Mult: {p['short_threshold_mult']:.2f}")
            print(f"    Short Max Hurst Adj: {p['short_max_hurst_adj']:.2f}")
            print(f"    Short Exit Mult: {p['short_exit_mult']:.2f}")
            if 'short_min_duration' in p:
                print(f"    Short Min Duration: {p['short_min_duration']}")
                print(f"    Short Min Amp Ratio: {p['short_min_amp_ratio']:.2f}")

    return result


if __name__ == "__main__":
    result = main()
