"""
Market Neutral Backtesting Framework for Extreme Hurst Signal
Implements beta and volatility controls for dollar/beta neutral portfolios
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
import warnings
import time
from scipy import stats
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS
from hurst_signal import (calculate_hurst, calculate_zscore,
                          calculate_atr, calculate_formation_duration, calculate_wave_amplitude)


def calculate_beta(returns: np.ndarray, market_returns: np.ndarray, window: int = 60) -> np.ndarray:
    """
    Calculate rolling beta relative to market
    Beta = Cov(stock, market) / Var(market)
    """
    n = len(returns)
    betas = np.ones(n)  # Default beta of 1

    for i in range(window, n):
        stock_window = returns[i-window:i]
        market_window = market_returns[i-window:i]

        # Remove NaN values
        valid = ~(np.isnan(stock_window) | np.isnan(market_window))
        if np.sum(valid) < 20:
            continue

        cov = np.cov(stock_window[valid], market_window[valid])[0, 1]
        var = np.var(market_window[valid])

        if var > 0:
            betas[i] = np.clip(cov / var, -2, 3)  # Clip extreme betas

    return betas


def calculate_rolling_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Calculate rolling volatility (annualized)"""
    n = len(returns)
    vol = np.full(n, np.nan)

    for i in range(window, n):
        vol[i] = np.std(returns[i-window:i]) * np.sqrt(252)

    # Fill early values with expanding window
    for i in range(1, window):
        if i >= 5:
            vol[i] = np.std(returns[:i]) * np.sqrt(252)

    # Replace any remaining NaN/0 with median
    median_vol = np.nanmedian(vol)
    if median_vol == 0 or np.isnan(median_vol):
        median_vol = 0.3  # Default 30% annualized vol
    vol = np.where(np.isnan(vol) | (vol == 0), median_vol, vol)

    return vol


def precompute_indicators_with_market(df: pd.DataFrame, market_returns: np.ndarray) -> Dict:
    """Precompute all indicators including beta and volatility"""
    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float) if 'High' in df.columns else close
    low = df['Low'].values.astype(float) if 'Low' in df.columns else close
    n = len(close)

    # Calculate returns
    returns = np.zeros(n)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]

    # Align market returns to same length
    if len(market_returns) > n:
        aligned_market = market_returns[-n:]
    elif len(market_returns) < n:
        aligned_market = np.zeros(n)
        aligned_market[-len(market_returns):] = market_returns
    else:
        aligned_market = market_returns

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

    # Formation duration and wave amplitude
    durations = {}
    for lb in [10, 15, 20]:
        z = zscores.get(lb, zscores[15])
        durations[lb] = calculate_formation_duration(close, z, threshold=1.0)

    wave_amp = calculate_wave_amplitude(close, lookback=20)

    # Beta and volatility for market neutral
    beta = calculate_beta(returns, aligned_market, window=60)
    volatility = calculate_rolling_volatility(returns, window=20)

    return {
        'close': close,
        'returns': returns,
        'n': n,
        'zscores': zscores,
        'hursts': hursts,
        'atr_ratio': atr_ratio,
        'durations': durations,
        'wave_amplitude': wave_amp,
        'beta': beta,
        'volatility': volatility,
    }


def market_neutral_backtest(indicators: Dict, base_threshold: float, ext_lb: int,
                            hurst_lb: int, max_hurst: float, min_exit: int, max_exit: int,
                            inertia_sens: float, duration_weight: float = 0.1,
                            amplitude_weight: float = 0.5,
                            short_threshold_mult: float = 1.0,
                            short_max_hurst_adj: float = 0.0,
                            short_exit_mult: float = 1.0,
                            short_min_duration: int = 0,
                            short_min_amp_ratio: float = 1.0,
                            target_vol: float = 0.15) -> Dict:
    """
    Market neutral backtest with beta and volatility controls

    Returns position-weighted results accounting for:
    - Volatility-adjusted position sizing (inverse volatility weighting)
    - Beta exposure tracking
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

    trades = []  # List of trade dicts

    min_idx = max(hurst_lb, ext_lb, 60, 25)  # Need 60 for beta calc
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

        # Skip if volatility or beta is invalid
        if np.isnan(vol) or vol <= 0 or np.isnan(b):
            continue

        # Volatility-adjusted position size (target_vol / actual_vol)
        position_size = min(target_vol / vol, 2.0)  # Cap at 2x leverage

        # Dynamic threshold
        hurst_factor = 0.5 + h * inertia_sens
        base_threshold_adj = base_threshold * hurst_factor * (0.8 + vol_adj * 0.4)

        # Inertia-based exit calculation
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
                'beta_exposure': position_size * b,  # Dollar-weighted beta
                'win': ret > 0,
            })

        # Short signal
        elif h < short_max_hurst and z > base_threshold_adj * short_threshold_mult:
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
                'beta_exposure': -position_size * b,  # Negative for shorts
                'win': ret > 0,
            })

    return {'trades': trades}


def aggregate_market_neutral_results(all_trades: List[Dict]) -> Dict:
    """
    Aggregate trades with market neutral analysis
    """
    if not all_trades:
        return {'long_hit_rate': 0, 'short_hit_rate': 0, 'total': 0}

    long_trades = [t for t in all_trades if t['direction'] == 'long']
    short_trades = [t for t in all_trades if t['direction'] == 'short']

    # Basic hit rates (unweighted)
    long_wins = sum(1 for t in long_trades if t['win'])
    short_wins = sum(1 for t in short_trades if t['win'])

    long_hit_rate = (long_wins / len(long_trades) * 100) if long_trades else 0
    short_hit_rate = (short_wins / len(short_trades) * 100) if short_trades else 0

    # Position-weighted returns
    long_weighted_returns = [t['return'] * t['position_size'] for t in long_trades]
    short_weighted_returns = [t['return'] * t['position_size'] for t in short_trades]

    # Beta exposures
    long_beta_exposure = sum(t['beta_exposure'] for t in long_trades) if long_trades else 0
    short_beta_exposure = sum(t['beta_exposure'] for t in short_trades) if short_trades else 0
    net_beta_exposure = long_beta_exposure + short_beta_exposure  # Should be near 0 for neutral

    # Average beta per side
    avg_long_beta = np.mean([t['beta'] for t in long_trades]) if long_trades else 0
    avg_short_beta = np.mean([t['beta'] for t in short_trades]) if short_trades else 0

    # Weighted hit rates (position-size weighted)
    if long_trades:
        total_long_size = sum(t['position_size'] for t in long_trades)
        weighted_long_wins = sum(t['position_size'] for t in long_trades if t['win'])
        weighted_long_hit_rate = (weighted_long_wins / total_long_size * 100)
    else:
        weighted_long_hit_rate = 0

    if short_trades:
        total_short_size = sum(t['position_size'] for t in short_trades)
        weighted_short_wins = sum(t['position_size'] for t in short_trades if t['win'])
        weighted_short_hit_rate = (weighted_short_wins / total_short_size * 100)
    else:
        weighted_short_hit_rate = 0

    # Total PnL analysis
    total_pnl = sum(long_weighted_returns) + sum(short_weighted_returns)
    long_pnl = sum(long_weighted_returns) if long_weighted_returns else 0
    short_pnl = sum(short_weighted_returns) if short_weighted_returns else 0

    return {
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'total': len(all_trades),
        'long_hit_rate': long_hit_rate,
        'short_hit_rate': short_hit_rate,
        'weighted_long_hit_rate': weighted_long_hit_rate,
        'weighted_short_hit_rate': weighted_short_hit_rate,
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'total_pnl': total_pnl,
        'net_beta_exposure': net_beta_exposure,
        'avg_long_beta': avg_long_beta,
        'avg_short_beta': avg_short_beta,
        'long_beta_exposure': long_beta_exposure,
        'short_beta_exposure': short_beta_exposure,
    }


def run_market_neutral_backtest(precomputed: Dict[str, Dict], **params) -> Dict:
    """Run market neutral backtest across all tickers"""
    all_trades = []

    for ticker, indicators in precomputed.items():
        result = market_neutral_backtest(indicators, **params)
        for trade in result['trades']:
            trade['ticker'] = ticker
        all_trades.extend(result['trades'])

    return aggregate_market_neutral_results(all_trades)


def beta_neutral_portfolio(all_trades: List[Dict]) -> Dict:
    """
    Construct a beta-neutral portfolio by reweighting positions
    Goal: Sum of (position_size * beta) = 0 for each time period
    """
    if not all_trades:
        return {'trades': [], 'metrics': {}}

    # Group trades by entry index (approximate time matching)
    from collections import defaultdict
    trades_by_time = defaultdict(list)

    for trade in all_trades:
        # Round to nearest 5 days for grouping
        time_bucket = trade['entry_idx'] // 5 * 5
        trades_by_time[time_bucket].append(trade)

    adjusted_trades = []

    for bucket, trades in trades_by_time.items():
        longs = [t for t in trades if t['direction'] == 'long']
        shorts = [t for t in trades if t['direction'] == 'short']

        if not longs or not shorts:
            # Can't be beta neutral without both sides
            adjusted_trades.extend(trades)
            continue

        # Calculate current beta exposures
        long_beta_exp = sum(t['position_size'] * t['beta'] for t in longs)
        short_beta_exp = sum(t['position_size'] * t['beta'] for t in shorts)  # Already negative in abs sense

        # To neutralize: adjust short positions so |short_beta_exp| = long_beta_exp
        # Scale factor for shorts
        if abs(short_beta_exp) > 0:
            target_short_beta = long_beta_exp
            current_short_beta = sum(t['position_size'] * t['beta'] for t in shorts)
            if current_short_beta != 0:
                scale = abs(target_short_beta / current_short_beta)
                scale = min(scale, 2.0)  # Cap scaling

                for t in shorts:
                    t = t.copy()
                    t['position_size'] *= scale
                    t['beta_exposure'] = -t['position_size'] * t['beta']
                    adjusted_trades.append(t)
                adjusted_trades.extend(longs)
            else:
                adjusted_trades.extend(trades)
        else:
            adjusted_trades.extend(trades)

    return {
        'trades': adjusted_trades,
        'metrics': aggregate_market_neutral_results(adjusted_trades)
    }


def optimize_market_neutral(precomputed: Dict[str, Dict], target: float = 60.0) -> Dict:
    """
    Optimize for market neutral with focus on both hit rates and balance
    """
    from multiprocessing import Pool, cpu_count
    import itertools

    print(f"\n{'='*60}")
    print(f"MARKET NEUTRAL OPTIMIZATION")
    print(f"Target: {target}% hit rate, balanced long/short, low net beta")
    print(f"{'='*60}\n")

    # Parameter grid - relaxed short requirements to get more trades
    base_thresholds = [3.0, 3.2, 3.5]
    ext_lbs = [15]
    hurst_lbs = [20]
    max_hursts = [0.35, 0.40, 0.45]
    min_exits = [3, 4, 5]
    max_exits = [6, 7, 8]
    inertia_sensitivities = [0.6, 0.8, 1.0]
    duration_weights = [0.02, 0.05]
    amplitude_weights = [0.35, 0.5]

    # Relaxed short requirements for more balanced trades
    short_threshold_mults = [0.9, 1.0, 1.1]
    short_max_hurst_adjs = [0.0, 0.02, 0.05]  # Allow higher Hurst for shorts
    short_exit_mults = [0.8, 1.0]
    short_min_durations = [2, 3, 4]  # Lower duration requirement
    short_min_amp_ratios = [1.0, 1.2, 1.3]  # Lower amplitude requirement

    all_configs = []
    for params in itertools.product(
        base_thresholds, ext_lbs, hurst_lbs, max_hursts, min_exits, max_exits,
        inertia_sensitivities, duration_weights, amplitude_weights,
        short_threshold_mults, short_max_hurst_adjs, short_exit_mults,
        short_min_durations, short_min_amp_ratios
    ):
        if params[4] >= params[5]:  # min_exit >= max_exit
            continue
        config = {
            'base_threshold': params[0],
            'ext_lb': params[1],
            'hurst_lb': params[2],
            'max_hurst': params[3],
            'min_exit': params[4],
            'max_exit': params[5],
            'inertia_sens': params[6],
            'duration_weight': params[7],
            'amplitude_weight': params[8],
            'short_threshold_mult': params[9],
            'short_max_hurst_adj': params[10],
            'short_exit_mult': params[11],
            'short_min_duration': params[12],
            'short_min_amp_ratio': params[13],
            'target_vol': 0.15,
        }
        all_configs.append(config)

    print(f"Testing {len(all_configs)} configurations...")

    best = None
    best_params = None
    best_score = 0

    for i, params in enumerate(tqdm(all_configs, desc="Optimizing")):
        result = run_market_neutral_backtest(precomputed, **params)

        # Scoring criteria for market neutral:
        # 1. Both hit rates should be high
        # 2. Trade count should be balanced (ratio close to 1)
        # 3. Net beta should be low

        if result['total'] < 100:
            continue
        if result['long_trades'] < 30 or result['short_trades'] < 30:
            continue

        # Balance ratio (1.0 = perfect balance)
        trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])

        # Penalize net beta exposure (normalized)
        beta_penalty = abs(result['net_beta_exposure']) / result['total'] if result['total'] > 0 else 1

        # Combined score
        min_hr = min(result['long_hit_rate'], result['short_hit_rate'])
        avg_hr = (result['long_hit_rate'] + result['short_hit_rate']) / 2

        # Score: prioritize hit rate, then balance, then low beta
        score = avg_hr * (0.5 + 0.5 * trade_ratio) * (1 - 0.1 * beta_penalty)

        if score > best_score:
            best_score = score
            best = result
            best_params = params.copy()
            print(f"\n  New best: Long {result['long_hit_rate']:.1f}% ({result['long_trades']}), "
                  f"Short {result['short_hit_rate']:.1f}% ({result['short_trades']}), "
                  f"Balance: {trade_ratio:.2f}, Net Beta: {result['net_beta_exposure']:.1f}")

    return {'result': best, 'params': best_params, 'score': best_score}


def main():
    """Main execution for market neutral backtest"""
    from data_fetcher import fetch_yahoo_data

    print("=" * 60)
    print("MARKET NEUTRAL EXTREME HURST SIGNAL")
    print("Beta and Volatility Controlled Portfolio")
    print("=" * 60)

    # Download market data (SPY) first
    print("\nDownloading market benchmark (SPY)...")
    spy = fetch_yahoo_data("SPY", period="2y")
    if spy is None or len(spy) == 0:
        print("Failed to download SPY data")
        return None

    spy_close = spy['Close'].values.flatten()
    market_returns = np.zeros(len(spy_close))
    market_returns[1:] = (spy_close[1:] - spy_close[:-1]) / spy_close[:-1]
    print(f"SPY data: {len(spy_close)} days")

    # Download ticker data
    data = download_all_data(TICKERS, period="2y")

    if len(data) < 50:
        print("Retrying data download...")
        time.sleep(3)
        data = download_all_data(TICKERS, period="2y")

    print(f"\nDownloaded {len(data)} tickers")

    # Precompute all indicators with market data
    print("\nPrecomputing indicators with beta/volatility...")
    precomputed = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            precomputed[ticker] = precompute_indicators_with_market(df, market_returns)
        except Exception as e:
            continue

    print(f"Precomputed indicators for {len(precomputed)} tickers")

    # Run optimization for market neutral
    opt_result = optimize_market_neutral(precomputed, target=60.0)

    if opt_result['result'] is None:
        print("\nNo valid configuration found!")
        return None

    result = opt_result['result']
    best_params = opt_result['params']

    print("\n" + "=" * 60)
    print("OPTIMIZED MARKET NEUTRAL RESULTS")
    print("=" * 60)
    print(f"Total Trades: {result['total']}")
    print(f"  Long Trades: {result['long_trades']}")
    print(f"  Short Trades: {result['short_trades']}")
    trade_ratio = min(result['long_trades'], result['short_trades']) / max(result['long_trades'], result['short_trades'])
    print(f"  Trade Balance Ratio: {trade_ratio:.2f}")
    print(f"\nHit Rates (Unweighted):")
    print(f"  Long:  {result['long_hit_rate']:.2f}%")
    print(f"  Short: {result['short_hit_rate']:.2f}%")
    print(f"\nHit Rates (Position-Size Weighted):")
    print(f"  Long:  {result['weighted_long_hit_rate']:.2f}%")
    print(f"  Short: {result['weighted_short_hit_rate']:.2f}%")
    print(f"\nPnL (Volatility-Adjusted):")
    print(f"  Long PnL:  {result['long_pnl']:.4f}")
    print(f"  Short PnL: {result['short_pnl']:.4f}")
    print(f"  Total PnL: {result['total_pnl']:.4f}")
    print(f"\nBeta Exposure:")
    print(f"  Avg Long Beta:  {result['avg_long_beta']:.2f}")
    print(f"  Avg Short Beta: {result['avg_short_beta']:.2f}")
    print(f"  Long Beta Exposure:  {result['long_beta_exposure']:.2f}")
    print(f"  Short Beta Exposure: {result['short_beta_exposure']:.2f}")
    print(f"  Net Beta Exposure:   {result['net_beta_exposure']:.2f}")
    print(f"  Net Beta per Trade:  {result['net_beta_exposure']/result['total']:.4f}")

    # Beta-neutral reweighting analysis
    print("\n" + "-" * 60)
    print("BETA-NEUTRAL REWEIGHTING ANALYSIS")
    print("-" * 60)

    all_trades = []
    for ticker, indicators in precomputed.items():
        ticker_result = market_neutral_backtest(indicators, **best_params)
        for trade in ticker_result['trades']:
            trade['ticker'] = ticker
        all_trades.extend(ticker_result['trades'])

    if all_trades:
        long_trades = [t for t in all_trades if t['direction'] == 'long']
        short_trades = [t for t in all_trades if t['direction'] == 'short']

        if long_trades and short_trades:
            # Calculate beta-neutral position sizes
            # Scale shorts to match long beta exposure
            long_beta_exp = sum(t['position_size'] * t['beta'] for t in long_trades)
            short_beta_exp = sum(t['position_size'] * t['beta'] for t in short_trades)

            if short_beta_exp > 0:
                short_scale = long_beta_exp / short_beta_exp
                short_scale = min(short_scale, 3.0)  # Cap scaling
            else:
                short_scale = 1.0

            print(f"\nTo achieve beta neutrality:")
            print(f"  Long beta exposure: {long_beta_exp:.2f}")
            print(f"  Short beta exposure (raw): {short_beta_exp:.2f}")
            print(f"  Required short scaling: {short_scale:.2f}x")

            # Recalculate with beta-neutral sizing
            bn_short_pnl = sum(t['return'] * t['position_size'] * short_scale for t in short_trades)
            bn_short_wins = sum(t['position_size'] * short_scale for t in short_trades if t['win'])
            bn_short_total = sum(t['position_size'] * short_scale for t in short_trades)

            bn_long_wins = sum(t['position_size'] for t in long_trades if t['win'])
            bn_long_total = sum(t['position_size'] for t in long_trades)
            bn_long_pnl = sum(t['return'] * t['position_size'] for t in long_trades)

            print(f"\nBeta-Neutral Weighted Hit Rates:")
            print(f"  Long:  {bn_long_wins/bn_long_total*100:.2f}%")
            print(f"  Short: {bn_short_wins/bn_short_total*100:.2f}%")
            print(f"\nBeta-Neutral PnL:")
            print(f"  Long:  {bn_long_pnl:.4f}")
            print(f"  Short: {bn_short_pnl:.4f}")
            print(f"  Total: {bn_long_pnl + bn_short_pnl:.4f}")

            # Verify neutrality
            adjusted_short_beta = short_beta_exp * short_scale
            print(f"\nAdjusted Net Beta Exposure: {long_beta_exp - adjusted_short_beta:.4f}")

    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETERS:")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return {'result': result, 'params': best_params}


if __name__ == "__main__":
    result = main()
