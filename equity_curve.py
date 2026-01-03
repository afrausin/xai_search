#!/usr/bin/env python3
"""
Equity Curve Visualization

Chart the cumulative return of the Hurst mean-reversion strategy.
"""

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from high_return_optimization import precompute_indicators, high_conviction_backtest, SECTOR_MAP

# Original 100 tickers (large-cap)
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

# Alternative 100 tickers
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

# Optimal parameters
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


def fetch_yahoo_data_extended(ticker: str, years: int = 10) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with extended history."""
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    period1 = int(start.timestamp())
    period2 = int(end.timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()

        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
            return None

        result = data["chart"]["result"][0]
        timestamps = result.get("timestamp", [])
        if not timestamps:
            return None

        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit="s"),
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        })

        df = df.set_index("Date")
        df = df.dropna()

        if len(df) < 200:
            return None

        return df

    except Exception:
        return None


def download_all_extended(tickers: list, years: int = 10) -> dict:
    """Download extended history for all tickers."""
    data = {}
    print(f"Downloading {years} years of history for {len(tickers)} tickers...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_yahoo_data_extended, t, years): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers)):
            ticker = futures[future]
            try:
                df = future.result()
                if df is not None:
                    data[ticker] = df
                time.sleep(0.1)
            except Exception:
                pass

    print(f"Successfully downloaded {len(data)} tickers")
    return data


def backtest_with_dates(indicators: dict, df: pd.DataFrame, ticker: str, **params) -> list:
    """
    Backtest that returns trades with entry/exit dates.
    Uses the original high_conviction_backtest logic but adds date info.
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
    min_idx = max(hurst_lb, ext_lb, 60, 30)
    wave_amp_mean = np.mean(wave_amp[wave_amp > 0]) if np.any(wave_amp > 0) else 1.0

    for i in range(min_idx, n - max_exit):
        h = hurst[i]
        z = zscore[i]
        vol = volatility[i]
        dur = duration[i]
        amp = wave_amp[i]
        amp_ratio = amp / wave_amp_mean if wave_amp_mean > 0 else 1.0

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

        vol_factor = min(vol / 0.25, 1.5)
        exit_bars = int(np.clip(min_exit + (max_exit - min_exit) * vol_factor * (1 - h), min_exit, max_exit))

        b = beta[i]
        b = np.clip(b, 0.3, 3.0) if not np.isnan(b) else 1.0

        position_size = target_volatility / vol if vol > 0 else 1.0
        position_size = np.clip(position_size, 0.25, 2.0)

        # Long signal
        if z < -z_threshold:
            if i + exit_bars >= n:
                continue
            entry = close[i]
            exit_price = close[i + exit_bars]
            ret = (exit_price - entry) / entry
            beta_adj_ret = ret / b

            trades.append({
                'direction': 'long',
                'ticker': ticker,
                'entry_date': dates[i],
                'exit_date': dates[i + exit_bars],
                'entry_idx': i,
                'exit_idx': i + exit_bars,
                'return': ret,
                'beta_adj_return': beta_adj_ret,
                'position_size': position_size,
                'beta': b,
            })

        # Short signal
        elif z > z_threshold:
            if i + exit_bars >= n:
                continue
            entry = close[i]
            exit_price = close[i + exit_bars]
            ret = (entry - exit_price) / entry
            beta_adj_ret = ret / b

            trades.append({
                'direction': 'short',
                'ticker': ticker,
                'entry_date': dates[i],
                'exit_date': dates[i + exit_bars],
                'entry_idx': i,
                'exit_idx': i + exit_bars,
                'return': ret,
                'beta_adj_return': beta_adj_ret,
                'position_size': position_size,
                'beta': b,
            })

    return trades


def main():
    print("=" * 70)
    print("EQUITY CURVE - HURST MEAN REVERSION STRATEGY")
    print("=" * 70)

    years = 10

    # Download SPY
    print(f"\nDownloading SPY ({years} years)...")
    market_df = fetch_yahoo_data_extended('SPY', years=years)
    if market_df is None:
        print("ERROR: Could not download SPY data")
        return
    print(f"SPY data: {len(market_df)} days from {market_df.index[0].date()} to {market_df.index[-1].date()}")

    # Download all tickers
    data = download_all_extended(ALL_TICKERS, years=years)

    # Precompute indicators using imported function
    print("\nPrecomputing indicators...")
    indicators_cache = {}
    df_cache = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            indicators_cache[ticker] = precompute_indicators(df, market_df)
            df_cache[ticker] = df
        except Exception as e:
            continue
    print(f"Precomputed for {len(indicators_cache)} tickers")

    # Run backtest with dates
    print("\nRunning backtest...")
    all_trades = []
    for ticker in tqdm(indicators_cache.keys(), desc="Backtesting"):
        indicators = indicators_cache[ticker]
        df = df_cache[ticker]
        trades = backtest_with_dates(indicators, df, ticker, **OPTIMAL_PARAMS)
        all_trades.extend(trades)

    print(f"Total trades: {len(all_trades)}")

    if not all_trades:
        print("No trades generated!")
        return

    # Sort trades by exit date
    all_trades.sort(key=lambda x: x['exit_date'])

    # Group trades by exit date
    trades_by_exit = {}
    for t in all_trades:
        exit_date = t['exit_date']
        if exit_date not in trades_by_exit:
            trades_by_exit[exit_date] = []
        trades_by_exit[exit_date].append(t)

    # Build equity curve (cumulative P&L, equal weight per trade)
    # Assume each trade uses 1% of capital (equal weight)
    trade_weight = 0.01  # 1% per trade

    cumulative_raw = 0.0
    cumulative_beta = 0.0
    equity_data = []

    for exit_date in sorted(trades_by_exit.keys()):
        trades = trades_by_exit[exit_date]

        # Add P&L from each trade (weighted)
        for t in trades:
            cumulative_raw += t['return'] * trade_weight
            cumulative_beta += t['beta_adj_return'] * trade_weight

        equity_data.append({
            'date': exit_date,
            'raw_equity': 1.0 + cumulative_raw,  # Starting at $1
            'beta_equity': 1.0 + cumulative_beta,
            'n_trades': len(trades),
            'avg_return': np.mean([t['return'] for t in trades]),
            'avg_beta_return': np.mean([t['beta_adj_return'] for t in trades]),
        })

    # Convert to DataFrame
    eq_df = pd.DataFrame(equity_data)
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')

    # Get SPY returns for comparison
    spy_start_date = eq_df.index[0]
    spy_end_date = eq_df.index[-1]
    spy_subset = market_df.loc[spy_start_date:spy_end_date]['Close']
    spy_normalized = spy_subset / spy_subset.iloc[0]

    # Calculate statistics
    total_raw_return = (eq_df['raw_equity'].iloc[-1] - 1) * 100
    total_beta_return = (eq_df['beta_equity'].iloc[-1] - 1) * 100
    spy_total_return = (spy_normalized.iloc[-1] - 1) * 100

    n_years = (eq_df.index[-1] - eq_df.index[0]).days / 365
    annual_raw = ((eq_df['raw_equity'].iloc[-1]) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    annual_beta = ((eq_df['beta_equity'].iloc[-1]) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    spy_annual = ((spy_normalized.iloc[-1]) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Period: {eq_df.index[0].date()} to {eq_df.index[-1].date()} ({n_years:.1f} years)")
    print(f"Total Trades: {len(all_trades)}")
    print(f"\nTotal Return:")
    print(f"  Strategy (Raw):       {total_raw_return:+.1f}%")
    print(f"  Strategy (Beta-Adj):  {total_beta_return:+.1f}%")
    print(f"  SPY Buy & Hold:       {spy_total_return:+.1f}%")
    print(f"\nAnnualized Return:")
    print(f"  Strategy (Raw):       {annual_raw:+.1f}%")
    print(f"  Strategy (Beta-Adj):  {annual_beta:+.1f}%")
    print(f"  SPY Buy & Hold:       {spy_annual:+.1f}%")

    # Create the chart
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Equity curves
    ax1 = axes[0]
    ax1.plot(eq_df.index, eq_df['raw_equity'], label=f'Strategy Raw ({total_raw_return:+.1f}%)', linewidth=2, color='blue')
    ax1.plot(eq_df.index, eq_df['beta_equity'], label=f'Strategy Beta-Adj ({total_beta_return:+.1f}%)', linewidth=2, color='green')
    ax1.plot(spy_normalized.index, spy_normalized.values, label=f'SPY Buy & Hold ({spy_total_return:+.1f}%)', linewidth=2, color='gray', alpha=0.7)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax1.set_title('Hurst Mean Reversion Strategy - Equity Curve (10 Years, 200 Stocks)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (starting at $1)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(eq_df.index[0], eq_df.index[-1])

    # Plot 2: Drawdown
    ax2 = axes[1]
    raw_peak = eq_df['raw_equity'].cummax()
    raw_dd = (eq_df['raw_equity'] - raw_peak) / raw_peak * 100
    beta_peak = eq_df['beta_equity'].cummax()
    beta_dd = (eq_df['beta_equity'] - beta_peak) / beta_peak * 100

    ax2.fill_between(eq_df.index, raw_dd, 0, alpha=0.3, color='blue', label='Raw Drawdown')
    ax2.fill_between(eq_df.index, beta_dd, 0, alpha=0.3, color='green', label='Beta-Adj Drawdown')
    ax2.set_title('Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(eq_df.index[0], eq_df.index[-1])

    # Plot 3: Trade count over time
    ax3 = axes[2]
    monthly_trades = eq_df['n_trades'].resample('M').sum()
    ax3.bar(monthly_trades.index, monthly_trades.values, width=20, alpha=0.7, color='purple')
    ax3.set_title('Monthly Trade Count', fontsize=12)
    ax3.set_ylabel('Number of Trades')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(eq_df.index[0], eq_df.index[-1])

    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: equity_curve.png")
    plt.close()

    # Yearly breakdown
    print("\n" + "=" * 70)
    print("YEARLY BREAKDOWN")
    print("=" * 70)

    # Group trades by year
    trades_by_year = {}
    for t in all_trades:
        year = t['exit_date'].year
        if year not in trades_by_year:
            trades_by_year[year] = []
        trades_by_year[year].append(t)

    print(f"\n{'Year':<6} | {'Trades':>7} | {'Avg Ret%':>10} | {'Avg Beta%':>10} | {'Year P&L%':>10} | {'Cumul P&L%':>10}")
    print("-" * 75)

    cumul_pnl = 0.0
    for year in sorted(trades_by_year.keys()):
        trades = trades_by_year[year]
        avg_ret = np.mean([t['return'] for t in trades]) * 100
        avg_beta = np.mean([t['beta_adj_return'] for t in trades]) * 100

        # Year P&L (sum of returns * weight)
        year_pnl = sum(t['beta_adj_return'] for t in trades) * trade_weight * 100
        cumul_pnl += year_pnl

        print(f"{year:<6} | {len(trades):>7} | {avg_ret:>10.2f} | {avg_beta:>10.2f} | {year_pnl:>+10.2f} | {cumul_pnl:>+10.2f}")


if __name__ == "__main__":
    main()
