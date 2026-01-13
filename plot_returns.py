#!/usr/bin/env python3
"""
Plot Returns Analysis

Visualize the distribution and characteristics of trade returns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from equity_curve import (
    fetch_yahoo_data_extended, download_all_extended,
    backtest_with_dates, calculate_z_weight,
    ALL_TICKERS, OPTIMAL_PARAMS
)
from high_return_optimization import precompute_indicators


def main():
    print("=" * 70)
    print("RETURNS ANALYSIS")
    print("=" * 70)

    years = 10

    # Download data
    print(f"\nDownloading SPY ({years} years)...")
    market_df = fetch_yahoo_data_extended('SPY', years=years)
    if market_df is None:
        print("ERROR: Could not download SPY data")
        return

    data = download_all_extended(ALL_TICKERS, years=years)

    # Precompute indicators
    print("\nPrecomputing indicators...")
    indicators_cache = {}
    df_cache = {}
    for ticker, df in tqdm(data.items(), desc="Computing"):
        try:
            indicators_cache[ticker] = precompute_indicators(df, market_df)
            df_cache[ticker] = df
        except:
            continue
    print(f"Precomputed for {len(indicators_cache)} tickers")

    # Run backtest
    print("\nRunning backtest...")
    all_trades = []
    for ticker in tqdm(indicators_cache.keys(), desc="Backtesting"):
        indicators = indicators_cache[ticker]
        df = df_cache[ticker]
        trades = backtest_with_dates(indicators, df, ticker, **OPTIMAL_PARAMS)
        all_trades.extend(trades)

    print(f"Total trades: {len(all_trades)}")

    # Add z_weight to trades
    for t in all_trades:
        t['z_weight'] = calculate_z_weight(t['z_score'], OPTIMAL_PARAMS['z_threshold'])

    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    df['exit_date'] = pd.to_datetime(df['exit_date'])
    df['return_pct'] = df['return'] * 100
    df['beta_adj_pct'] = df['beta_adj_return'] * 100
    df['weighted_return'] = df['beta_adj_return'] * df['z_weight'] * 100

    # Create figure with multiple plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Return distribution (histogram)
    ax1 = axes[0, 0]
    ax1.hist(df['beta_adj_pct'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=df['beta_adj_pct'].mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean: {df["beta_adj_pct"].mean():.2f}%')
    ax1.set_xlabel('Beta-Adjusted Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Trade Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Long vs Short returns
    ax2 = axes[0, 1]
    long_rets = df[df['direction'] == 'long']['beta_adj_pct']
    short_rets = df[df['direction'] == 'short']['beta_adj_pct']
    ax2.hist(long_rets, bins=30, alpha=0.6, color='blue', label=f'Long (n={len(long_rets)}, μ={long_rets.mean():.2f}%)')
    ax2.hist(short_rets, bins=30, alpha=0.6, color='red', label=f'Short (n={len(short_rets)}, μ={short_rets.mean():.2f}%)')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Beta-Adjusted Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Long vs Short Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Returns by Z-Score
    ax3 = axes[0, 2]
    df['z_abs'] = df['z_score'].abs()
    scatter = ax3.scatter(df['z_abs'], df['beta_adj_pct'], c=df['z_weight'], cmap='RdYlGn', alpha=0.6, s=30)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('|Z-Score|')
    ax3.set_ylabel('Beta-Adjusted Return (%)')
    ax3.set_title('Returns vs Z-Score (color = weight)')
    plt.colorbar(scatter, ax=ax3, label='Z-Weight')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative returns over time
    ax4 = axes[1, 0]
    df_sorted = df.sort_values('exit_date')
    df_sorted['cumul_return'] = df_sorted['beta_adj_pct'].cumsum() * 0.01  # 1% weight per trade
    df_sorted['cumul_weighted'] = (df_sorted['beta_adj_pct'] * df_sorted['z_weight']).cumsum() * 0.01
    ax4.plot(df_sorted['exit_date'], df_sorted['cumul_return'], label='Unweighted', color='green', linewidth=2)
    ax4.plot(df_sorted['exit_date'], df_sorted['cumul_weighted'], label='Z-Weighted', color='orange', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.set_title('Cumulative Returns Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Returns by year (box plot)
    ax5 = axes[1, 1]
    df['year'] = df['exit_date'].dt.year
    years_list = sorted(df['year'].unique())
    box_data = [df[df['year'] == y]['beta_adj_pct'].values for y in years_list]
    bp = ax5.boxplot(box_data, labels=years_list, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Beta-Adjusted Return (%)')
    ax5.set_title('Return Distribution by Year')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Win rate by Z-score quintile
    ax6 = axes[1, 2]
    df['z_quintile'] = pd.qcut(df['z_abs'], 5, labels=['Q1\n(3.0)', 'Q2', 'Q3', 'Q4', 'Q5\n(3.4+)'])
    df['win'] = df['beta_adj_return'] > 0

    quintile_stats = df.groupby('z_quintile').agg({
        'beta_adj_pct': 'mean',
        'win': 'mean'
    })

    x = range(5)
    hit_rates = quintile_stats['win'].values * 100
    avg_returns = quintile_stats['beta_adj_pct'].values

    bars = ax6.bar(x, hit_rates, color=['green' if h > 50 else 'red' for h in hit_rates], alpha=0.7, edgecolor='black')
    ax6.axhline(y=50, color='black', linestyle='--', linewidth=1, label='50% (random)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Q1\n(|z|≈3.0)', 'Q2', 'Q3', 'Q4', 'Q5\n(|z|>3.4)'])
    ax6.set_xlabel('Z-Score Quintile')
    ax6.set_ylabel('Hit Rate (%)')
    ax6.set_title('Hit Rate by Z-Score Quintile')

    # Add return labels on bars
    for i, (bar, ret) in enumerate(zip(bars, avg_returns)):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ret:.1f}%', ha='center', va='bottom', fontsize=9)
    ax6.set_ylim(0, 100)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('returns_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: returns_analysis.png")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nOverall:")
    print(f"  Total Trades: {len(df)}")
    print(f"  Mean Return: {df['beta_adj_pct'].mean():.2f}%")
    print(f"  Median Return: {df['beta_adj_pct'].median():.2f}%")
    print(f"  Std Dev: {df['beta_adj_pct'].std():.2f}%")
    print(f"  Hit Rate: {(df['beta_adj_return'] > 0).mean() * 100:.1f}%")
    print(f"  Sharpe (per trade): {df['beta_adj_pct'].mean() / df['beta_adj_pct'].std():.3f}")

    print(f"\nBy Direction:")
    for direction in ['long', 'short']:
        sub = df[df['direction'] == direction]
        print(f"  {direction.upper()}: n={len(sub)}, mean={sub['beta_adj_pct'].mean():.2f}%, hit={((sub['beta_adj_return'] > 0).mean() * 100):.1f}%")

    print(f"\nBy Z-Score Quintile:")
    for q in df['z_quintile'].unique():
        sub = df[df['z_quintile'] == q]
        hit = (sub['beta_adj_return'] > 0).mean() * 100
        avg_z = sub['z_abs'].mean()
        print(f"  {q}: n={len(sub)}, mean={sub['beta_adj_pct'].mean():.2f}%, hit={hit:.1f}%, avg|z|={avg_z:.2f}")


if __name__ == "__main__":
    main()
