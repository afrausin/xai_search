"""
High Return Optimization
Target: >2.5% average return per trade
Strategy: Only take highest conviction signals with larger expected moves
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
import itertools
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS, fetch_yahoo_data
from hurst_signal import (calculate_hurst, calculate_zscore,
                          calculate_atr, calculate_formation_duration, calculate_wave_amplitude)

SECTOR_MAP = {
    'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech', 'NVDA': 'tech',
    'META': 'tech', 'TSLA': 'tech', 'AVGO': 'tech', 'ORCL': 'tech', 'ADBE': 'tech',
    'CRM': 'tech', 'AMD': 'tech', 'INTC': 'tech', 'QCOM': 'tech', 'TXN': 'tech',
    'CSCO': 'tech', 'IBM': 'tech', 'AMAT': 'tech', 'MU': 'tech', 'LRCX': 'tech',
    'JPM': 'finance', 'BAC': 'finance', 'WFC': 'finance', 'GS': 'finance', 'MS': 'finance',
    'C': 'finance', 'BLK': 'finance', 'SCHW': 'finance', 'AXP': 'finance', 'USB': 'finance',
    'UNH': 'healthcare', 'JNJ': 'healthcare', 'PFE': 'healthcare', 'MRK': 'healthcare',
    'ABBV': 'healthcare', 'LLY': 'healthcare', 'TMO': 'healthcare', 'ABT': 'healthcare',
    'DHR': 'healthcare', 'BMY': 'healthcare',
    'WMT': 'consumer', 'PG': 'consumer', 'KO': 'consumer', 'PEP': 'consumer', 'COST': 'consumer',
    'MCD': 'consumer', 'NKE': 'consumer', 'SBUX': 'consumer', 'HD': 'consumer', 'LOW': 'consumer',
    'CAT': 'industrial', 'BA': 'industrial', 'GE': 'industrial', 'HON': 'industrial',
    'UPS': 'industrial', 'RTX': 'industrial', 'LMT': 'industrial', 'DE': 'industrial',
    'MMM': 'industrial', 'FDX': 'industrial',
    'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy', 'SLB': 'energy', 'EOG': 'energy',
    'MPC': 'energy', 'PSX': 'energy', 'VLO': 'energy', 'OXY': 'energy', 'KMI': 'energy',
    'DIS': 'communication', 'CMCSA': 'communication', 'NFLX': 'communication', 'T': 'communication',
    'VZ': 'communication', 'TMUS': 'communication', 'CHTR': 'communication', 'EA': 'communication',
    'WBD': 'communication', 'PARA': 'communication',
    'LIN': 'materials', 'APD': 'materials', 'SHW': 'materials', 'ECL': 'materials',
    'FCX': 'materials', 'NEM': 'materials', 'NUE': 'materials', 'DOW': 'materials',
    'DD': 'materials', 'PPG': 'materials',
    'AMT': 'reits', 'PLD': 'reits', 'CCI': 'reits', 'EQIX': 'reits', 'SPG': 'reits',
    'NEE': 'utilities', 'DUK': 'utilities', 'SO': 'utilities', 'D': 'utilities', 'AEP': 'utilities',
}


def precompute_indicators(df: pd.DataFrame, market_df: pd.DataFrame) -> Dict:
    """Precompute all indicators"""
    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float) if 'High' in df.columns else close
    low = df['Low'].values.astype(float) if 'Low' in df.columns else close
    n = len(close)

    zscores = {}
    for lb in [10, 15, 20, 25, 30]:
        zscores[lb] = calculate_zscore(close, lb)

    hursts = {}
    for hurst_lb in [15, 20, 25, 30, 40]:
        hursts[hurst_lb] = calculate_hurst(close, hurst_lb)

    atr = calculate_atr(high, low, close)
    atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)

    durations = {}
    for lb in [10, 15, 20]:
        z = zscores.get(lb, zscores[15])
        durations[lb] = calculate_formation_duration(close, z, threshold=1.0)

    wave_amp = calculate_wave_amplitude(close, lookback=20)

    # Beta and volatility
    market_close = market_df['Close'].values.astype(float)
    min_len = min(len(close), len(market_close))
    stock_rets = np.diff(np.log(close[:min_len]))
    market_rets = np.diff(np.log(market_close[:min_len]))

    beta = np.full(n, 1.0)
    volatility = np.full(n, 0.2)
    lookback = 60

    for i in range(lookback, min_len - 1):
        s_r = stock_rets[i-lookback:i]
        m_r = market_rets[i-lookback:i]
        cov = np.cov(s_r, m_r)[0, 1]
        var = np.var(m_r)
        if var > 0:
            beta[i+1] = cov / var
        volatility[i+1] = np.std(s_r) * np.sqrt(252)

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
    }


def high_conviction_backtest(indicators: Dict, ticker: str,
                              # Entry thresholds - VERY strict
                              z_threshold: float = 4.0,  # High z-score requirement
                              max_hurst: float = 0.35,   # Only strong mean reversion
                              min_volatility: float = 0.20,  # Only volatile stocks
                              # Duration requirements
                              min_duration: int = 5,  # Extended for longer
                              min_amp_ratio: float = 1.5,  # Large wave amplitude
                              # Exit parameters
                              min_exit: int = 5,
                              max_exit: int = 15,  # Longer holds for larger moves
                              # Lookbacks
                              ext_lb: int = 20,
                              hurst_lb: int = 25,
                              ) -> List[Dict]:
    """
    High conviction backtest - only take the best setups
    """
    close = indicators['close']
    n = indicators['n']
    zscore = indicators['zscores'].get(ext_lb, indicators['zscores'][20])
    hurst = indicators['hursts'].get(hurst_lb, indicators['hursts'][25])
    atr_ratio = indicators['atr_ratio']
    duration = indicators['durations'].get(min(ext_lb, 20), indicators['durations'][15])
    wave_amp = indicators['wave_amplitude']
    volatility = indicators['volatility']

    trades = []
    min_idx = max(hurst_lb, ext_lb, 60, 30)
    wave_amp_mean = np.mean(wave_amp[wave_amp > 0]) if np.any(wave_amp > 0) else 1.0

    sector = SECTOR_MAP.get(ticker, 'unknown')

    for i in range(min_idx, n - max_exit):
        h = hurst[i]
        z = zscore[i]
        vol = volatility[i]
        dur = duration[i]
        amp = wave_amp[i]
        amp_ratio = amp / wave_amp_mean if wave_amp_mean > 0 else 1.0

        if np.isnan(vol) or vol <= 0:
            continue

        # Skip if volatility too low (small moves expected)
        if vol < min_volatility:
            continue

        # Skip if Hurst too high (not mean reverting)
        if h >= max_hurst:
            continue

        # Skip if not extended long enough
        if dur < min_duration:
            continue

        # Skip if wave amplitude not large enough
        if amp_ratio < min_amp_ratio:
            continue

        # Dynamic exit based on volatility - higher vol = longer hold for bigger move
        vol_factor = min(vol / 0.25, 1.5)  # Scale based on volatility
        exit_bars = int(np.clip(min_exit + (max_exit - min_exit) * vol_factor * (1 - h), min_exit, max_exit))

        # Long signal - very negative z-score
        if z < -z_threshold:
            if i + exit_bars >= n:
                continue
            entry = close[i]
            exit_price = close[i + exit_bars]
            ret = (exit_price - entry) / entry

            trades.append({
                'direction': 'long',
                'ticker': ticker,
                'sector': sector,
                'return': ret,
                'hold_days': exit_bars,
                'z_score': z,
                'hurst': h,
                'volatility': vol,
                'win': ret > 0,
            })

        # Short signal - very positive z-score
        elif z > z_threshold:
            if i + exit_bars >= n:
                continue
            entry = close[i]
            exit_price = close[i + exit_bars]
            ret = (entry - exit_price) / entry

            trades.append({
                'direction': 'short',
                'ticker': ticker,
                'sector': sector,
                'return': ret,
                'hold_days': exit_bars,
                'z_score': z,
                'hurst': h,
                'volatility': vol,
                'win': ret > 0,
            })

    return trades


# Global precomputed for multiprocessing
_precomputed = None


def init_worker(precomputed):
    global _precomputed
    _precomputed = precomputed


def test_config(params):
    """Test a single configuration"""
    global _precomputed
    all_trades = []

    for ticker, indicators in _precomputed.items():
        trades = high_conviction_backtest(indicators, ticker, **params)
        all_trades.extend(trades)

    # Need minimum trades for statistical significance
    if len(all_trades) < 30:
        return None, None, -999

    long_trades = [t for t in all_trades if t['direction'] == 'long']
    short_trades = [t for t in all_trades if t['direction'] == 'short']

    if len(long_trades) < 10 or len(short_trades) < 10:
        return None, None, -999

    # Calculate average returns
    long_rets = [t['return'] for t in long_trades]
    short_rets = [t['return'] for t in short_trades]
    all_rets = [t['return'] for t in all_trades]

    avg_return = np.mean(all_rets) * 100
    long_avg = np.mean(long_rets) * 100
    short_avg = np.mean(short_rets) * 100

    long_hr = sum(1 for t in long_trades if t['win']) / len(long_trades) * 100
    short_hr = sum(1 for t in short_trades if t['win']) / len(short_trades) * 100

    result = {
        'total_trades': len(all_trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'avg_return': avg_return,
        'long_avg_return': long_avg,
        'short_avg_return': short_avg,
        'long_hit_rate': long_hr,
        'short_hit_rate': short_hr,
    }

    return result, params, avg_return


def main():
    print("=" * 70)
    print("HIGH RETURN OPTIMIZATION")
    print("Target: >2.5% average return per trade")
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

    # Parameter grid - focused on high conviction
    z_thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]  # Very strict
    max_hursts = [0.30, 0.35, 0.40]  # Low Hurst = strong mean reversion
    min_volatilities = [0.15, 0.20, 0.25, 0.30]  # Only volatile stocks
    min_durations = [3, 5, 7, 10]  # Extended state duration
    min_amp_ratios = [1.0, 1.3, 1.5, 2.0]  # Wave amplitude requirement
    min_exits = [5, 7, 10]  # Longer holds
    max_exits = [10, 15, 20]  # Allow time for mean reversion
    ext_lbs = [15, 20, 25]
    hurst_lbs = [20, 25, 30]

    # Generate configurations
    all_configs = []
    for z_thresh, max_h, min_vol, min_dur, min_amp, min_ex, max_ex, ext_lb, h_lb in itertools.product(
        z_thresholds, max_hursts, min_volatilities, min_durations,
        min_amp_ratios, min_exits, max_exits, ext_lbs, hurst_lbs
    ):
        if min_ex >= max_ex:
            continue
        params = {
            'z_threshold': z_thresh,
            'max_hurst': max_h,
            'min_volatility': min_vol,
            'min_duration': min_dur,
            'min_amp_ratio': min_amp,
            'min_exit': min_ex,
            'max_exit': max_ex,
            'ext_lb': ext_lb,
            'hurst_lb': h_lb,
        }
        all_configs.append(params)

    print(f"\nTesting {len(all_configs)} configurations...")

    best_result = None
    best_params = None
    best_avg = -999

    # Process in batches
    batch_size = 500
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(precomputed,)) as pool:
        for batch_start in tqdm(range(0, len(all_configs), batch_size), desc="Batches"):
            batch = all_configs[batch_start:batch_start + batch_size]
            results = pool.map(test_config, batch)

            for result, params, avg_ret in results:
                if result is None:
                    continue

                if avg_ret > best_avg:
                    best_avg = avg_ret
                    best_result = result
                    best_params = params.copy()
                    print(f"\n  New best: Avg {avg_ret:.3f}% (L:{result['long_avg_return']:.2f}%, S:{result['short_avg_return']:.2f}%)")
                    print(f"    Trades: {result['total_trades']} (L:{result['long_trades']}, S:{result['short_trades']})")
                    print(f"    Hit rates: L:{result['long_hit_rate']:.1f}%, S:{result['short_hit_rate']:.1f}%")

                    if avg_ret >= 2.5:
                        print(f"\n✓ TARGET ACHIEVED: {avg_ret:.3f}% avg return!")

    if best_result:
        print("\n" + "=" * 70)
        print("FINAL BEST RESULT")
        print("=" * 70)
        print(f"Average Return per Trade: {best_result['avg_return']:.4f}%")
        print(f"  Long Avg: {best_result['long_avg_return']:.4f}%")
        print(f"  Short Avg: {best_result['short_avg_return']:.4f}%")
        print(f"\nTrades: {best_result['total_trades']} (L:{best_result['long_trades']}, S:{best_result['short_trades']})")
        print(f"Hit Rates: L:{best_result['long_hit_rate']:.2f}%, S:{best_result['short_hit_rate']:.2f}%")

        print(f"\nBest Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    return best_result, best_params


if __name__ == "__main__":
    main()
