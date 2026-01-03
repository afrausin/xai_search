"""
Return Analysis - Calculate average return per position
Uses the actual sector-based backtest with best parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

from data_fetcher import download_all_data, TICKERS, fetch_yahoo_data
from hurst_signal import (calculate_hurst, calculate_zscore,
                          calculate_atr, calculate_formation_duration, calculate_wave_amplitude)

# Sector mapping
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

# Best parameters from sector optimization (883 longs, 213 shorts)
BEST_PARAMS = {
    'base_threshold': 2.0,  # From optimization - lower threshold for more trades
    'ext_lb': 15,
    'hurst_lb': 20,
    'max_hurst': 0.45,  # From optimization
    'min_exit': 3,
    'max_exit': 8,
    'inertia_sens': 0.6,
    'duration_weight': 0.02,
    'amplitude_weight': 0.35,
    'short_threshold_mult': 1.0,
    'short_max_hurst_adj': -0.02,
    'short_exit_mult': 1.0,
    'short_min_duration': 3,
    'short_min_amp_ratio': 1.2,
    'target_vol': 0.15,
    'short_only_high_beta': False,
    'short_exclude_defensive': True,  # Key: exclude defensive sectors
}


def precompute_indicators_with_market(df: pd.DataFrame, market_df: pd.DataFrame) -> Dict:
    """Precompute all indicators including beta and volatility"""
    close = df['Close'].values.astype(float)
    high = df['High'].values.astype(float) if 'High' in df.columns else close
    low = df['Low'].values.astype(float) if 'Low' in df.columns else close
    n = len(close)

    # Z-scores
    zscores = {}
    for lb in [10, 15, 20, 25]:
        zscores[lb] = calculate_zscore(close, lb)

    # Hurst
    hursts = {}
    for hurst_lb in [15, 20, 25, 30]:
        hursts[hurst_lb] = calculate_hurst(close, hurst_lb)

    # ATR
    atr = calculate_atr(high, low, close)
    atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)

    # Durations
    durations = {}
    for lb in [10, 15, 20]:
        z = zscores.get(lb, zscores[15])
        durations[lb] = calculate_formation_duration(close, z, threshold=1.0)

    wave_amp = calculate_wave_amplitude(close, lookback=20)

    # Beta calculation
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


HIGH_BETA_SECTORS = {'tech', 'energy', 'finance', 'materials', 'communication'}
DEFENSIVE_SECTORS = {'healthcare', 'consumer', 'utilities', 'reits'}


def backtest_with_returns(indicators: Dict, ticker: str, params: Dict) -> List[Dict]:
    """Backtest with full return tracking"""
    close = indicators['close']
    n = indicators['n']

    ext_lb = params['ext_lb']
    hurst_lb = params['hurst_lb']

    zscore = indicators['zscores'].get(ext_lb, indicators['zscores'][15])
    hurst = indicators['hursts'].get(hurst_lb, indicators['hursts'][20])
    atr_ratio = indicators['atr_ratio']
    duration = indicators['durations'].get(ext_lb, indicators['durations'][15])
    wave_amp = indicators['wave_amplitude']
    beta = indicators['beta']
    volatility = indicators['volatility']

    base_threshold = params['base_threshold']
    max_hurst = params['max_hurst']
    min_exit = params['min_exit']
    max_exit = params['max_exit']
    inertia_sens = params['inertia_sens']
    duration_weight = params['duration_weight']
    amplitude_weight = params['amplitude_weight']
    short_threshold_mult = params['short_threshold_mult']
    short_max_hurst_adj = params['short_max_hurst_adj']
    short_exit_mult = params['short_exit_mult']
    short_min_duration = params['short_min_duration']
    short_min_amp_ratio = params['short_min_amp_ratio']
    target_vol = params['target_vol']
    short_exclude_defensive = params.get('short_exclude_defensive', True)

    sector = SECTOR_MAP.get(ticker, 'unknown')
    is_defensive = sector in DEFENSIVE_SECTORS

    trades = []
    min_idx = max(hurst_lb, ext_lb, 60, 25)
    base_bars = (min_exit + max_exit) / 2
    short_max_hurst = max_hurst + short_max_hurst_adj
    wave_amp_mean = np.mean(wave_amp[wave_amp > 0]) if np.any(wave_amp > 0) else 1.0

    # Allow shorts?
    allow_shorts = True
    if short_exclude_defensive and is_defensive:
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
                'ticker': ticker,
                'sector': sector,
                'entry_price': entry,
                'exit_price': exit_price,
                'return': ret,
                'sized_return': ret * position_size,
                'hold_days': exit_bars,
                'position_size': position_size,
                'beta': b,
                'win': ret > 0,
            })

        # Short signal
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
                'ticker': ticker,
                'sector': sector,
                'entry_price': entry,
                'exit_price': exit_price,
                'return': ret,
                'sized_return': ret * position_size,
                'hold_days': short_exit_bars,
                'position_size': position_size,
                'beta': b,
                'win': ret > 0,
            })

    return trades


def main():
    print("=" * 70)
    print("RETURN ANALYSIS - AVERAGE RETURN PER POSITION")
    print("Using sector-based best parameters")
    print("=" * 70)

    # Download data
    data = download_all_data(TICKERS, period="2y")
    print(f"\nDownloaded {len(data)} tickers")

    # Download market data
    print("Downloading SPY for beta calculation...")
    market_df = fetch_yahoo_data('SPY', period='2y')

    # Precompute indicators
    print("\nPrecomputing indicators...")
    precomputed = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            precomputed[ticker] = precompute_indicators_with_market(df, market_df)
        except:
            continue

    print(f"Precomputed for {len(precomputed)} tickers")

    # Run backtest
    print("\nRunning backtest...")
    all_trades = []
    for ticker, indicators in tqdm(precomputed.items(), desc="Backtesting"):
        trades = backtest_with_returns(indicators, ticker, BEST_PARAMS)
        all_trades.extend(trades)

    # Separate longs and shorts
    long_trades = [t for t in all_trades if t['direction'] == 'long']
    short_trades = [t for t in all_trades if t['direction'] == 'short']

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # LONG TRADES
    print(f"\n{'='*30} LONG TRADES {'='*30}")
    print(f"Total Trades: {len(long_trades)}")

    if long_trades:
        long_wins = [t for t in long_trades if t['win']]
        long_losses = [t for t in long_trades if not t['win']]
        long_rets = [t['return'] for t in long_trades]
        long_sized = [t['sized_return'] for t in long_trades]

        print(f"Hit Rate: {len(long_wins)/len(long_trades)*100:.2f}%")
        print(f"\nRaw Returns (before position sizing):")
        print(f"  Average: {np.mean(long_rets)*100:.4f}%")
        print(f"  Median: {np.median(long_rets)*100:.4f}%")
        print(f"  Std Dev: {np.std(long_rets)*100:.4f}%")
        print(f"  Min/Max: {np.min(long_rets)*100:.2f}% / {np.max(long_rets)*100:.2f}%")

        print(f"\nPosition-Sized Returns:")
        print(f"  Average: {np.mean(long_sized)*100:.4f}%")
        print(f"  Median: {np.median(long_sized)*100:.4f}%")

        if long_wins:
            win_rets = [t['return'] for t in long_wins]
            print(f"\nWinning Trades ({len(long_wins)}):")
            print(f"  Average Win: {np.mean(win_rets)*100:.4f}%")

        if long_losses:
            loss_rets = [t['return'] for t in long_losses]
            print(f"\nLosing Trades ({len(long_losses)}):")
            print(f"  Average Loss: {np.mean(loss_rets)*100:.4f}%")

        total_win = sum(t['return'] for t in long_wins) if long_wins else 0
        total_loss = abs(sum(t['return'] for t in long_losses)) if long_losses else 0.001
        print(f"\nProfit Factor: {total_win/total_loss:.2f}")

        hold_times = [t['hold_days'] for t in long_trades]
        print(f"Average Hold: {np.mean(hold_times):.1f} days")

    # SHORT TRADES
    print(f"\n{'='*30} SHORT TRADES {'='*29}")
    print(f"Total Trades: {len(short_trades)}")

    if short_trades:
        short_wins = [t for t in short_trades if t['win']]
        short_losses = [t for t in short_trades if not t['win']]
        short_rets = [t['return'] for t in short_trades]
        short_sized = [t['sized_return'] for t in short_trades]

        print(f"Hit Rate: {len(short_wins)/len(short_trades)*100:.2f}%")
        print(f"\nRaw Returns (before position sizing):")
        print(f"  Average: {np.mean(short_rets)*100:.4f}%")
        print(f"  Median: {np.median(short_rets)*100:.4f}%")
        print(f"  Std Dev: {np.std(short_rets)*100:.4f}%")
        print(f"  Min/Max: {np.min(short_rets)*100:.2f}% / {np.max(short_rets)*100:.2f}%")

        print(f"\nPosition-Sized Returns:")
        print(f"  Average: {np.mean(short_sized)*100:.4f}%")
        print(f"  Median: {np.median(short_sized)*100:.4f}%")

        if short_wins:
            win_rets = [t['return'] for t in short_wins]
            print(f"\nWinning Trades ({len(short_wins)}):")
            print(f"  Average Win: {np.mean(win_rets)*100:.4f}%")

        if short_losses:
            loss_rets = [t['return'] for t in short_losses]
            print(f"\nLosing Trades ({len(short_losses)}):")
            print(f"  Average Loss: {np.mean(loss_rets)*100:.4f}%")

        total_win = sum(t['return'] for t in short_wins) if short_wins else 0
        total_loss = abs(sum(t['return'] for t in short_losses)) if short_losses else 0.001
        print(f"\nProfit Factor: {total_win/total_loss:.2f}")

        hold_times = [t['hold_days'] for t in short_trades]
        print(f"Average Hold: {np.mean(hold_times):.1f} days")

    # COMBINED
    print(f"\n{'='*30} COMBINED {'='*33}")
    print(f"Total Trades: {len(all_trades)}")
    all_wins = [t for t in all_trades if t['win']]
    print(f"Overall Hit Rate: {len(all_wins)/len(all_trades)*100:.2f}%")

    all_rets = [t['return'] for t in all_trades]
    all_sized = [t['sized_return'] for t in all_trades]

    print(f"\nRaw Average Return: {np.mean(all_rets)*100:.4f}%")
    print(f"Position-Sized Avg: {np.mean(all_sized)*100:.4f}%")
    print(f"Total Raw Return: {sum(all_rets)*100:.2f}%")
    print(f"Total Sized Return: {sum(all_sized)*100:.2f}%")

    # Annualized
    avg_hold = np.mean([t['hold_days'] for t in all_trades])
    trades_per_year = 250 / avg_hold * (len(all_trades) / 2)  # 2-year period
    print(f"\nEstimates (2-year backtest period):")
    print(f"  Trades per year: ~{trades_per_year:.0f}")
    print(f"  Average hold: {avg_hold:.1f} days")
    print(f"  Raw annual return: {np.mean(all_rets) * trades_per_year * 100:.2f}%")
    print(f"  Sized annual return: {np.mean(all_sized) * trades_per_year * 100:.2f}%")

    # By sector
    print(f"\n{'='*30} BY SECTOR {'='*32}")
    print(f"{'Sector':<15} {'Dir':<6} {'Trades':>7} {'HR':>8} {'Avg Ret':>10} {'Win Avg':>10} {'Loss Avg':>10}")
    print("-" * 70)

    for direction in ['long', 'short']:
        dir_trades = [t for t in all_trades if t['direction'] == direction]
        sector_data = {}
        for t in dir_trades:
            s = t['sector']
            if s not in sector_data:
                sector_data[s] = []
            sector_data[s].append(t)

        for sector in sorted(sector_data.keys()):
            trades = sector_data[sector]
            wins = [t for t in trades if t['win']]
            losses = [t for t in trades if not t['win']]
            hr = len(wins) / len(trades) * 100
            avg_ret = np.mean([t['return'] for t in trades]) * 100
            win_avg = np.mean([t['return'] for t in wins]) * 100 if wins else 0
            loss_avg = np.mean([t['return'] for t in losses]) * 100 if losses else 0
            print(f"{sector:<15} {direction:<6} {len(trades):>7} {hr:>7.1f}% {avg_ret:>9.3f}% {win_avg:>9.3f}% {loss_avg:>9.3f}%")


if __name__ == "__main__":
    main()
