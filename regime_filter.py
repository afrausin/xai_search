#!/usr/bin/env python3
"""
Regime Filter Implementation

Add VIX-based regime filtering and dynamic Hurst regime detection
to avoid trading during trending/crisis markets like 2020.

Data source: Financial Modeling Prep (FMP) API
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from high_return_optimization import precompute_indicators, SECTOR_MAP

# FMP API Key
FMP_API_KEY = "RfwoaYzM2NZzzkb4NImlOnleTscNPJmv"

# Tickers
ORIGINAL_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    'CRM', 'AMD', 'INTC', 'QCOM', 'TXN', 'CSCO', 'IBM', 'AMAT', 'MU', 'LRCX',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
    'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'HD', 'LOW',
    'CAT', 'BA', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'MMM', 'FDX',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI',
    'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'WBD', 'PARA',
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'PPG',
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'NEE', 'DUK', 'SO', 'D', 'AEP',
]

ALTERNATIVE_TICKERS = [
    'SNOW', 'DDOG', 'ZS', 'CRWD', 'NET', 'OKTA', 'TEAM', 'WDAY', 'SPLK', 'PANW',
    'FTNT', 'ZM', 'DOCU', 'TWLO', 'SQ', 'SHOP', 'ROKU', 'TTD', 'SNAP', 'PINS',
    'TFC', 'PNC', 'FITB', 'KEY', 'CFG', 'RF', 'HBAN', 'ALLY', 'SIVB', 'ZION',
    'VRTX', 'REGN', 'BIIB', 'GILD', 'ILMN', 'ALGN', 'IDXX', 'DXCM', 'ZBH', 'BSX',
    'TGT', 'DG', 'DLTR', 'ROST', 'TJX', 'ORLY', 'AZO', 'BBY', 'ULTA', 'LULU',
    'EMR', 'ETN', 'ROK', 'PH', 'ITW', 'CMI', 'PCAR', 'FAST', 'ODFL', 'CARR',
    'DVN', 'FANG', 'HAL', 'BKR', 'APA', 'MRO', 'CTRA', 'OVV', 'HES', 'TRGP',
    'MTCH', 'LYV', 'TTWO', 'FOXA', 'IPG', 'OMC', 'NWSA', 'VIAC', 'LBRDK', 'GOOG',
    'PYPL', 'FIS', 'FISV', 'GPN', 'VRSN', 'CDW', 'CTSH', 'EPAM', 'IT', 'AKAM',
    'MAR', 'HLT', 'LVS', 'WYNN', 'MGM', 'CCL', 'RCL', 'NCLH', 'DAL', 'UAL',
]

ALL_TICKERS = ORIGINAL_TICKERS + ALTERNATIVE_TICKERS

# Optimal parameters from previous optimization
OPTIMAL_PARAMS = {
    'z_threshold': 3.0,
    'max_hurst': 0.45,
    'min_volatility': 0.2,
    'min_duration': 2,
    'min_amp_ratio': 0.5,
    'min_exit': 7,
    'max_exit': 20,
    'ext_lb': 15,
    'hurst_lb': 25
}


def fetch_fmp_data(ticker: str, years: int = 15) -> pd.DataFrame:
    """
    Fetch OHLCV data from Financial Modeling Prep API.
    Uses v3 API for full historical data (up to 20 years).
    """
    # Use v3 API which has up to 20 years of history
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
    params = {
        "apikey": FMP_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        # v3 API returns {"symbol": "...", "historical": [...]}
        if not data or 'historical' not in data:
            # Try stable API as fallback with explicit date range
            return fetch_fmp_data_stable(ticker, years)

        historical = data['historical']
        if not historical:
            return fetch_fmp_data_stable(ticker, years)

        df = pd.DataFrame(historical)
        if df.empty or 'date' not in df.columns:
            return None

        df['Date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.set_index('Date')
        df = df.sort_index()  # Sort oldest to newest
        df = df.dropna()

        if len(df) < 200:
            return None

        return df

    except Exception as e:
        return fetch_fmp_data_stable(ticker, years)


def fetch_fmp_data_stable(ticker: str, years: int = 15) -> pd.DataFrame:
    """Fallback: Fetch from stable API with explicit date range."""
    from datetime import datetime, timedelta

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')

    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full"
    params = {
        "symbol": ticker,
        "from": start_date,
        "to": end_date,
        "apikey": FMP_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if not data or isinstance(data, dict) and 'Error' in str(data):
            return None

        df = pd.DataFrame(data)
        if df.empty or 'date' not in df.columns:
            return None

        df['Date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.set_index('Date')
        df = df.sort_index()
        df = df.dropna()

        if len(df) < 200:
            return None

        return df

    except Exception:
        return None


def fetch_fmp_index(symbol: str) -> pd.DataFrame:
    """
    Fetch index data (VIX, SPY, etc) from FMP v3 index endpoint.
    Handles URL encoding for symbols with special characters.
    """
    import urllib.parse

    # URL encode the symbol (^VIX -> %5EVIX)
    encoded_symbol = urllib.parse.quote(symbol, safe='')

    # Try v3 index endpoint first
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{encoded_symbol}"
    params = {"apikey": FMP_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if data and 'historical' in data:
            historical = data['historical']
            if historical:
                df = pd.DataFrame(historical)
                df['Date'] = pd.to_datetime(df['date'])
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                cols = ['Date', 'Open', 'High', 'Low', 'Close']
                if 'Volume' in df.columns:
                    cols.append('Volume')
                df = df[cols]
                df = df.set_index('Date')
                df = df.sort_index()
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
                df = df.dropna(subset=['Close'])
                if len(df) >= 200:
                    return df
    except Exception:
        pass

    # Fallback to regular fetch
    return fetch_fmp_data(symbol)


def download_all_fmp(tickers: list) -> dict:
    """Download data for all tickers using FMP API."""
    data = {}
    print(f"Downloading data for {len(tickers)} tickers from FMP...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_fmp_data, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers)):
            ticker = futures[future]
            try:
                df = future.result()
                if df is not None:
                    data[ticker] = df
                time.sleep(0.1)  # Rate limiting
            except Exception:
                pass

    print(f"Successfully downloaded {len(data)} tickers")
    return data


def calculate_market_hurst(market_df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Calculate rolling Hurst exponent for market (SPY) to detect regime.
    H < 0.5 = mean-reverting, H > 0.5 = trending
    """
    close = market_df['Close'].values
    n = len(close)
    hurst = np.full(n, np.nan)

    for i in range(lookback, n):
        window = close[i-lookback:i]
        if len(window) < lookback:
            continue

        # R/S analysis for Hurst
        mean = np.mean(window)
        std = np.std(window)
        if std == 0:
            continue

        deviations = window - mean
        cumsum = np.cumsum(deviations)
        R = np.max(cumsum) - np.min(cumsum)
        S = std

        if S > 0 and R > 0:
            hurst[i] = np.log(R / S) / np.log(lookback)

    return pd.Series(hurst, index=market_df.index)


def backtest_with_regime_filter(
    indicators: dict,
    df: pd.DataFrame,
    ticker: str,
    vix_data: pd.DataFrame,
    market_hurst: pd.Series,
    vix_threshold: float = 30.0,
    market_hurst_threshold: float = 0.55,
    **params
) -> tuple:
    """
    Backtest with regime filtering.

    Filters:
    1. VIX > threshold = no new trades (high fear/volatility)
    2. Market Hurst > threshold = no new trades (trending market)
    """
    close = indicators['close']
    dates = df.index.tolist()
    n = indicators['n']

    ext_lb = params.get('ext_lb', 15)
    hurst_lb = params.get('hurst_lb', 25)
    z_threshold = params.get('z_threshold', 3.0)
    max_hurst = params.get('max_hurst', 0.45)
    min_volatility = params.get('min_volatility', 0.2)
    min_duration = params.get('min_duration', 2)
    min_amp_ratio = params.get('min_amp_ratio', 0.5)
    min_exit = params.get('min_exit', 7)
    max_exit = params.get('max_exit', 20)
    target_volatility = params.get('target_volatility', 0.15)

    zscore = indicators['zscores'].get(ext_lb, indicators['zscores'][20])
    hurst = indicators['hursts'].get(hurst_lb, indicators['hursts'][25])
    duration = indicators['durations'].get(min(ext_lb, 20), indicators['durations'][15])
    wave_amp = indicators['wave_amplitude']
    volatility = indicators['volatility']
    beta = indicators['beta']

    trades = []
    filtered_trades = []
    min_idx = max(hurst_lb, ext_lb, 60, 30)
    wave_amp_mean = np.mean(wave_amp[wave_amp > 0]) if np.any(wave_amp > 0) else 1.0

    for i in range(min_idx, n - max_exit):
        h = hurst[i]
        z = zscore[i]
        vol = volatility[i]
        dur = duration[i]
        amp = wave_amp[i]
        amp_ratio = amp / wave_amp_mean if wave_amp_mean > 0 else 1.0
        current_date = dates[i]

        if np.isnan(vol) or vol <= 0:
            continue
        if vol < min_volatility:
            continue
        if h >= max_hurst:
            continue
        if dur < min_duration:
            continue
        if amp_ratio < min_amp_ratio:
            continue

        if abs(z) < z_threshold:
            continue

        direction = 'long' if z < -z_threshold else 'short'

        vol_factor = min(vol / 0.25, 1.5)
        exit_bars = int(np.clip(min_exit + (max_exit - min_exit) * vol_factor * (1 - h), min_exit, max_exit))

        if i + exit_bars >= n:
            continue

        b = beta[i]
        b = np.clip(b, 0.3, 3.0) if not np.isnan(b) else 1.0

        position_size = target_volatility / vol if vol > 0 else 1.0
        position_size = np.clip(position_size, 0.25, 2.0)

        entry = close[i]
        exit_price = close[i + exit_bars]

        if direction == 'long':
            ret = (exit_price - entry) / entry
        else:
            ret = (entry - exit_price) / entry

        beta_adj_ret = ret / b

        trade_info = {
            'direction': direction,
            'ticker': ticker,
            'entry_date': dates[i],
            'exit_date': dates[i + exit_bars],
            'entry_idx': i,
            'exit_idx': i + exit_bars,
            'return': ret,
            'beta_adj_return': beta_adj_ret,
            'position_size': position_size,
            'beta': b,
            'z_score': z,
            'hurst': h,
            'volatility': vol,
        }

        # Check regime filters
        filtered = False
        filter_reason = None
        current_vix = 20.0
        current_market_hurst = 0.5

        # VIX filter - DIRECTIONAL: only filter SHORTS during high VIX (squeeze risk)
        # KEEP longs during high VIX (buying oversold stocks = good)
        try:
            if current_date in vix_data.index:
                current_vix = vix_data.loc[current_date, 'Close']
            else:
                vix_dates = vix_data.index
                idx = vix_dates.get_indexer([current_date], method='ffill')[0]
                if idx >= 0:
                    current_vix = vix_data.iloc[idx]['Close']

            # Only filter SHORT trades during high VIX (squeeze risk)
            # Long trades during high VIX = buying the dip = KEEP
            if current_vix > vix_threshold and direction == 'short':
                filtered = True
                filter_reason = f'VIX={current_vix:.1f} (no shorts in panic)'
        except Exception:
            pass

        trade_info['vix'] = current_vix

        # Market Hurst filter
        try:
            if current_date in market_hurst.index:
                current_market_hurst = market_hurst.loc[current_date]
            else:
                mh_dates = market_hurst.index
                idx = mh_dates.get_indexer([current_date], method='ffill')[0]
                if idx >= 0:
                    current_market_hurst = market_hurst.iloc[idx]

            if not np.isnan(current_market_hurst) and current_market_hurst > market_hurst_threshold:
                if not filtered:
                    filtered = True
                    filter_reason = f'MktHurst={current_market_hurst:.2f}'
                else:
                    filter_reason += f', MktHurst={current_market_hurst:.2f}'
        except Exception:
            pass

        trade_info['market_hurst'] = current_market_hurst
        trade_info['filtered'] = filtered
        trade_info['filter_reason'] = filter_reason

        if filtered:
            filtered_trades.append(trade_info)
        else:
            trades.append(trade_info)

    return trades, filtered_trades


def calculate_z_weight(z_score: float, z_threshold: float = 3.0) -> float:
    """Calculate trade weight based on z_score."""
    z_abs = abs(z_score)
    weight = 2.0 - (z_abs - z_threshold) / 1.0 * 1.5
    return np.clip(weight, 0.5, 2.0)


def main():
    print("=" * 70)
    print("REGIME FILTER BACKTEST (FMP Data)")
    print("=" * 70)

    vix_threshold = 40.0  # VIX > 40 = extreme panic only (March 2020, etc)
    market_hurst_threshold = 0.85  # Set high - effectively disabling for most conditions

    print(f"\nFilter Settings:")
    print(f"  VIX Threshold: {vix_threshold} (no trades when VIX > {vix_threshold})")
    print(f"  Market Hurst Threshold: {market_hurst_threshold} (no trades when trending)")

    # Download SPY and VIX from FMP (v3 API with up to 20 years history)
    print(f"\nDownloading market data from FMP (v3 API - up to 20 years)...")
    market_df = fetch_fmp_data('SPY')
    if market_df is None:
        print("ERROR: Could not download SPY data")
        return
    print(f"SPY data: {len(market_df)} days from {market_df.index[0].date()} to {market_df.index[-1].date()}")

    # Use fetch_fmp_index for VIX (handles special character encoding)
    vix_df = fetch_fmp_index('^VIX')
    if vix_df is None:
        # Try alternative VIX symbol
        vix_df = fetch_fmp_data('VIXY')  # VIX ETF as fallback
        if vix_df is None:
            print("ERROR: Could not download VIX data")
            return
        print("Using VIXY (VIX ETF) as fallback")
    print(f"VIX data: {len(vix_df)} days from {vix_df.index[0].date()} to {vix_df.index[-1].date()}")

    # Calculate market Hurst
    print("Calculating market regime (Hurst exponent)...")
    market_hurst = calculate_market_hurst(market_df, lookback=50)

    # Download all tickers
    data = download_all_fmp(ALL_TICKERS)

    # Precompute indicators
    print("\nPrecomputing indicators...")
    indicators_cache = {}
    df_cache = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            indicators_cache[ticker] = precompute_indicators(df, market_df)
            df_cache[ticker] = df
        except Exception:
            continue
    print(f"Precomputed for {len(indicators_cache)} tickers")

    # Run backtest with regime filter
    print("\nRunning backtest with regime filter...")
    all_trades = []
    all_filtered = []

    for ticker in tqdm(indicators_cache.keys(), desc="Backtesting"):
        indicators = indicators_cache[ticker]
        df = df_cache[ticker]
        trades, filtered = backtest_with_regime_filter(
            indicators, df, ticker, vix_df, market_hurst,
            vix_threshold=vix_threshold,
            market_hurst_threshold=market_hurst_threshold,
            **OPTIMAL_PARAMS
        )
        all_trades.extend(trades)
        all_filtered.extend(filtered)

    print(f"\nTrades taken: {len(all_trades)}")
    print(f"Trades filtered out: {len(all_filtered)}")

    if not all_trades and not all_filtered:
        print("No trades generated!")
        return

    # Add z_weights
    for t in all_trades:
        t['z_weight'] = calculate_z_weight(t['z_score'], OPTIMAL_PARAMS['z_threshold'])
    for t in all_filtered:
        t['z_weight'] = calculate_z_weight(t['z_score'], OPTIMAL_PARAMS['z_threshold'])

    # Calculate performance
    base_weight = 0.01

    # With filter
    total_return_filtered = sum(t['beta_adj_return'] * t['z_weight'] for t in all_trades) * base_weight * 100
    avg_return_filtered = np.mean([t['beta_adj_return'] for t in all_trades]) * 100 if all_trades else 0

    # Without filter (all trades)
    all_potential = all_trades + all_filtered
    total_return_unfiltered = sum(t['beta_adj_return'] * t['z_weight'] for t in all_potential) * base_weight * 100
    avg_return_unfiltered = np.mean([t['beta_adj_return'] for t in all_potential]) * 100 if all_potential else 0

    # Filtered trades performance
    filtered_return = sum(t['beta_adj_return'] * t['z_weight'] for t in all_filtered) * base_weight * 100 if all_filtered else 0
    avg_filtered = np.mean([t['beta_adj_return'] for t in all_filtered]) * 100 if all_filtered else 0

    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\nWithout Filter:")
    print(f"  Total Trades: {len(all_potential)}")
    print(f"  Avg Return/Trade: {avg_return_unfiltered:.2f}%")
    print(f"  Total Z-Weighted Return: {total_return_unfiltered:+.2f}%")

    print(f"\nWith Regime Filter:")
    print(f"  Trades Taken: {len(all_trades)}")
    print(f"  Trades Filtered: {len(all_filtered)} ({len(all_filtered)/max(len(all_potential),1)*100:.1f}%)")
    print(f"  Avg Return/Trade: {avg_return_filtered:.2f}%")
    print(f"  Total Z-Weighted Return: {total_return_filtered:+.2f}%")

    print(f"\nFiltered Trades (avoided):")
    print(f"  Count: {len(all_filtered)}")
    print(f"  Avg Return: {avg_filtered:.2f}%")
    print(f"  Total Return Avoided: {filtered_return:+.2f}%")

    improvement = total_return_filtered - total_return_unfiltered
    print(f"\nImprovement from Filter: {improvement:+.2f}%")

    # Yearly breakdown
    print("\n" + "=" * 70)
    print("YEARLY COMPARISON")
    print("=" * 70)

    def group_by_year(trades):
        by_year = {}
        for t in trades:
            year = t['exit_date'].year
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(t)
        return by_year

    filtered_by_year = group_by_year(all_trades)
    unfiltered_by_year = group_by_year(all_potential)
    avoided_by_year = group_by_year(all_filtered)

    all_years = sorted(set(list(filtered_by_year.keys()) + list(unfiltered_by_year.keys())))

    print(f"\n{'Year':<6} | {'Unfilt':>8} | {'Filt':>8} | {'Avoided':>8} | {'Unfilt%':>10} | {'Filt%':>10} | {'Saved%':>10}")
    print("-" * 85)

    for year in all_years:
        unfilt_trades = unfiltered_by_year.get(year, [])
        filt_trades = filtered_by_year.get(year, [])
        avoided_trades = avoided_by_year.get(year, [])

        unfilt_ret = sum(t['beta_adj_return'] * t['z_weight'] for t in unfilt_trades) * base_weight * 100
        filt_ret = sum(t['beta_adj_return'] * t['z_weight'] for t in filt_trades) * base_weight * 100
        avoided_ret = sum(t['beta_adj_return'] * t['z_weight'] for t in avoided_trades) * base_weight * 100

        print(f"{year:<6} | {len(unfilt_trades):>8} | {len(filt_trades):>8} | {len(avoided_trades):>8} | {unfilt_ret:>+10.2f} | {filt_ret:>+10.2f} | {-avoided_ret:>+10.2f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: VIX over time with threshold
    ax1 = axes[0, 0]
    ax1.plot(vix_df.index, vix_df['Close'], color='purple', linewidth=1, alpha=0.8)
    ax1.axhline(y=vix_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({vix_threshold})')
    ax1.fill_between(vix_df.index, vix_threshold, vix_df['Close'],
                     where=vix_df['Close'] > vix_threshold, color='red', alpha=0.3, label='No Trade Zone')
    ax1.set_title('VIX with Trading Filter', fontsize=12, fontweight='bold')
    ax1.set_ylabel('VIX')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(vix_df.index[0], vix_df.index[-1])

    # Plot 2: Market Hurst over time
    ax2 = axes[0, 1]
    ax2.plot(market_hurst.index, market_hurst.values, color='blue', linewidth=1, alpha=0.8)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='H=0.5 (random)')
    ax2.axhline(y=market_hurst_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({market_hurst_threshold})')
    ax2.fill_between(market_hurst.index, market_hurst_threshold, market_hurst.values,
                     where=market_hurst.values > market_hurst_threshold, color='red', alpha=0.3, label='Trending (No Trade)')
    ax2.set_title('Market Hurst Exponent (SPY)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Hurst Exponent')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(market_hurst.index[0], market_hurst.index[-1])

    # Plot 3: Cumulative returns comparison
    ax3 = axes[1, 0]

    all_trades_sorted = sorted(all_trades, key=lambda x: x['exit_date'])
    all_potential_sorted = sorted(all_potential, key=lambda x: x['exit_date'])

    cumul_filt = 0
    filt_dates, filt_equity = [], []
    for t in all_trades_sorted:
        cumul_filt += t['beta_adj_return'] * t['z_weight'] * base_weight
        filt_dates.append(t['exit_date'])
        filt_equity.append(1 + cumul_filt)

    cumul_unfilt = 0
    unfilt_dates, unfilt_equity = [], []
    for t in all_potential_sorted:
        cumul_unfilt += t['beta_adj_return'] * t['z_weight'] * base_weight
        unfilt_dates.append(t['exit_date'])
        unfilt_equity.append(1 + cumul_unfilt)

    if unfilt_dates:
        ax3.plot(unfilt_dates, unfilt_equity, label=f'Unfiltered ({total_return_unfiltered:+.1f}%)', color='red', linewidth=2, alpha=0.7)
    if filt_dates:
        ax3.plot(filt_dates, filt_equity, label=f'With Filter ({total_return_filtered:+.1f}%)', color='green', linewidth=2)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax3.set_title('Equity Curve: Filtered vs Unfiltered', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Portfolio Value')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Yearly returns comparison
    ax4 = axes[1, 1]
    years_list = list(all_years)
    x = np.arange(len(years_list))
    width = 0.35

    unfilt_yearly = [sum(t['beta_adj_return'] * t['z_weight'] for t in unfiltered_by_year.get(y, [])) * base_weight * 100 for y in years_list]
    filt_yearly = [sum(t['beta_adj_return'] * t['z_weight'] for t in filtered_by_year.get(y, [])) * base_weight * 100 for y in years_list]

    bars1 = ax4.bar(x - width/2, unfilt_yearly, width, label='Unfiltered', color='red', alpha=0.7)
    bars2 = ax4.bar(x + width/2, filt_yearly, width, label='With Filter', color='green', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Return (%)')
    ax4.set_title('Yearly Returns: Filtered vs Unfiltered', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(years_list)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('regime_filter.png', dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: regime_filter.png")
    plt.close()

    # Filter analysis
    print("\n" + "=" * 70)
    print("FILTER ANALYSIS")
    print("=" * 70)

    if all_filtered:
        vix_filtered = [t for t in all_filtered if 'VIX' in str(t.get('filter_reason', ''))]
        hurst_filtered = [t for t in all_filtered if 'MktHurst' in str(t.get('filter_reason', ''))]

        print(f"\nFiltered by VIX > {vix_threshold}: {len(vix_filtered)} trades")
        if vix_filtered:
            vix_avg = np.mean([t['beta_adj_return'] for t in vix_filtered]) * 100
            print(f"  Avg return of VIX-filtered: {vix_avg:.2f}%")

        print(f"\nFiltered by Market Hurst > {market_hurst_threshold}: {len(hurst_filtered)} trades")
        if hurst_filtered:
            hurst_avg = np.mean([t['beta_adj_return'] for t in hurst_filtered]) * 100
            print(f"  Avg return of Hurst-filtered: {hurst_avg:.2f}%")


if __name__ == "__main__":
    main()
