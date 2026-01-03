"""
Backtesting Framework for Extreme Hurst Signal
Ultra-fast version with precomputed indicators
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


def precompute_indicators(df: pd.DataFrame) -> Dict:
    """Precompute all indicators once per ticker"""
    close = df['Close'].values.astype(float)
    n = len(close)

    # Precompute z-scores at different lookbacks
    zscores = {}
    for lb in [10, 15, 20, 25, 30]:
        roll_mean = pd.Series(close).rolling(lb).mean().values
        roll_std = pd.Series(close).rolling(lb).std().values
        roll_std = np.where(roll_std > 0, roll_std, 1e-10)
        zscores[lb] = (close - roll_mean) / roll_std

    # Precompute RSI
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gains).rolling(14, min_periods=1).mean().values
    avg_loss = pd.Series(losses).rolling(14, min_periods=1).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    # Precompute Hurst at different lookbacks (sparse for speed)
    hursts = {}
    for hurst_lb in [20, 30, 40, 50]:
        hurst = np.full(n, 0.5)
        log_p = np.log(np.maximum(close, 1e-10))
        for i in range(hurst_lb, n, 10):  # Every 10 bars
            seg = log_p[max(0, i-hurst_lb):i]
            if len(seg) >= 15:
                lags = [2, 4, 8]
                taus = []
                for lag in lags:
                    if lag < len(seg):
                        diff = seg[lag:] - seg[:-lag]
                        std = np.std(diff)
                        if std > 0:
                            taus.append((lag, std))
                if len(taus) >= 2:
                    log_lags = np.log([t[0] for t in taus])
                    log_taus = np.log([t[1] for t in taus])
                    slope, _, _, _, _ = stats.linregress(log_lags, log_taus)
                    hurst[i] = np.clip(slope, 0.0, 1.0)
        # Forward fill
        for i in range(1, n):
            if hurst[i] == 0.5 and hurst[i-1] != 0.5 and i % 10 != 0:
                hurst[i] = hurst[i-1]
        hursts[hurst_lb] = hurst

    return {
        'close': close,
        'n': n,
        'zscores': zscores,
        'rsi': rsi,
        'hursts': hursts,
    }


def fast_backtest(indicators: Dict, threshold: float, exit_bars: int,
                  ext_lb: int, hurst_lb: int, max_hurst: float,
                  use_rsi: bool, rsi_os: float, rsi_ob: float) -> Dict:
    """Ultra-fast backtest using precomputed indicators"""
    close = indicators['close']
    n = indicators['n']
    zscore = indicators['zscores'].get(ext_lb, indicators['zscores'][20])
    rsi = indicators['rsi']
    hurst = indicators['hursts'].get(hurst_lb, indicators['hursts'][30])

    long_trades = []
    short_trades = []

    min_idx = max(hurst_lb, ext_lb, 20)

    for i in range(min_idx, n - exit_bars):
        z = zscore[i]
        h = hurst[i]
        r = rsi[i]

        # Long signal
        if z < -threshold and h < max_hurst:
            if not use_rsi or r < rsi_os:
                entry = close[i]
                exit_price = close[i + exit_bars]
                ret = (exit_price - entry) / entry
                long_trades.append(ret > 0)

        # Short signal
        elif z > threshold and h < max_hurst:
            if not use_rsi or r > rsi_ob:
                entry = close[i]
                exit_price = close[i + exit_bars]
                ret = (entry - exit_price) / entry
                short_trades.append(ret > 0)

    return {
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_wins': sum(long_trades),
        'short_wins': sum(short_trades),
    }


def run_fast_backtest(precomputed: Dict[str, Dict], **params) -> Dict:
    """Run backtest across all tickers with precomputed indicators"""
    total_long = 0
    total_short = 0
    long_wins = 0
    short_wins = 0

    for ticker, indicators in precomputed.items():
        result = fast_backtest(indicators, **params)
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


def optimize_signal(precomputed: Dict[str, Dict], target: float = 60.0) -> Dict:
    """Optimize signal parameters"""
    print(f"\n{'='*60}")
    print(f"OPTIMIZING FOR {target}% HIT RATE ON BOTH LONGS AND SHORTS")
    print(f"{'='*60}\n")

    best = None
    best_params = None
    best_min = 0

    # Parameter grid
    thresholds = [1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0]
    ext_lbs = [10, 15, 20, 25]
    hurst_lbs = [20, 30, 40]
    max_hursts = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    exit_bars_list = [2, 3, 5, 7, 10]
    rsi_os_list = [20, 25, 30, 35]
    rsi_ob_list = [65, 70, 75, 80]

    total_configs = (len(thresholds) * len(ext_lbs) * len(hurst_lbs) *
                    len(max_hursts) * len(exit_bars_list) * 2 *
                    len(rsi_os_list) * len(rsi_ob_list))

    print(f"Testing {total_configs} configurations...")

    configs_tested = 0

    for threshold in tqdm(thresholds, desc="Threshold"):
        for ext_lb in ext_lbs:
            for hurst_lb in hurst_lbs:
                for max_hurst in max_hursts:
                    for exit_bars in exit_bars_list:
                        for use_rsi in [True, False]:
                            for rsi_os in rsi_os_list:
                                for rsi_ob in rsi_ob_list:
                                    params = {
                                        'threshold': threshold,
                                        'exit_bars': exit_bars,
                                        'ext_lb': ext_lb,
                                        'hurst_lb': hurst_lb,
                                        'max_hurst': max_hurst,
                                        'use_rsi': use_rsi,
                                        'rsi_os': rsi_os,
                                        'rsi_ob': rsi_ob,
                                    }

                                    result = run_fast_backtest(precomputed, **params)
                                    configs_tested += 1

                                    if result['total'] < 20:
                                        continue

                                    min_hr = min(result['long_hit_rate'], result['short_hit_rate'])

                                    if min_hr > best_min:
                                        best_min = min_hr
                                        best = result
                                        best_params = params.copy()

                                        if min_hr >= target:
                                            print(f"\n✓ TARGET ACHIEVED!")
                                            print(f"  Long: {result['long_hit_rate']:.1f}%")
                                            print(f"  Short: {result['short_hit_rate']:.1f}%")
                                            print(f"  Trades: {result['total']}")
                                            return {'result': best, 'params': best_params}

    if best:
        print(f"\nBest achieved after {configs_tested} configs:")
        print(f"  Long: {best['long_hit_rate']:.1f}%, Short: {best['short_hit_rate']:.1f}%")
        print(f"  Trades: {best['total']}")

    return {'result': best, 'params': best_params}


def main():
    """Main execution"""
    print("=" * 60)
    print("EXTREME HURST TRADING SIGNAL - BACKTEST")
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

        print(f"\nOptimal Parameters:")
        print(f"  Extension Threshold: {p['threshold']:.2f}")
        print(f"  Extension Lookback: {p['ext_lb']}")
        print(f"  Hurst Lookback: {p['hurst_lb']}")
        print(f"  Max Hurst: {p['max_hurst']:.2f}")
        print(f"  Exit Bars: {p['exit_bars']}")
        print(f"  Use RSI: {p['use_rsi']}")
        print(f"  RSI Oversold: {p['rsi_os']}")
        print(f"  RSI Overbought: {p['rsi_ob']}")

    return result


if __name__ == "__main__":
    result = main()
