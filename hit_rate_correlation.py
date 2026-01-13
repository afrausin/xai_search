#!/usr/bin/env python3
"""
Hit Rate Correlation Analysis

Find which variables best predict whether a trade wins.
Use this to weight position sizes.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from high_return_optimization import precompute_indicators, high_conviction_backtest
from data_fetcher import download_all_data, TICKERS, fetch_yahoo_data

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


def main():
    print("=" * 70)
    print("HIT RATE CORRELATION ANALYSIS")
    print("Finding which variables predict winning trades")
    print("=" * 70)

    # Download data
    print("\nDownloading market data...")
    market_df = fetch_yahoo_data('SPY', period='2y')
    data = download_all_data(TICKERS, period='2y')

    # Precompute indicators
    print("\nPrecomputing indicators...")
    indicators_cache = {}
    for ticker, df in data.items():
        try:
            indicators_cache[ticker] = precompute_indicators(df, market_df)
        except:
            continue
    print(f"Precomputed for {len(indicators_cache)} tickers")

    # Run backtest to get trades with all variables
    print("\nRunning backtest...")
    all_trades = []
    for ticker, indicators in indicators_cache.items():
        trades = high_conviction_backtest(indicators, ticker, **OPTIMAL_PARAMS)
        all_trades.extend(trades)

    print(f"Total trades: {len(all_trades)}")

    if not all_trades:
        print("No trades!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_trades)

    # Add win indicator
    df['win'] = df['beta_adj_return'] > 0

    # Variables to analyze
    variables = ['z_score', 'hurst', 'volatility', 'beta', 'hold_days', 'position_size']

    # Make z_score absolute (magnitude matters for both long/short)
    df['z_score_abs'] = df['z_score'].abs()

    print("\n" + "=" * 70)
    print("CORRELATION WITH WIN (beta_adj_return > 0)")
    print("=" * 70)

    correlations = []
    for var in variables + ['z_score_abs']:
        if var not in df.columns:
            continue

        # Point-biserial correlation (binary outcome vs continuous)
        valid = df[[var, 'win']].dropna()
        if len(valid) < 10:
            continue

        corr, pval = stats.pointbiserialr(valid['win'], valid[var])
        correlations.append({
            'variable': var,
            'correlation': corr,
            'p_value': pval,
            'significant': pval < 0.05
        })

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    print(f"\n{'Variable':<15} | {'Correlation':>12} | {'P-Value':>10} | {'Significant':>10}")
    print("-" * 60)
    for c in correlations:
        sig = "YES" if c['significant'] else "no"
        print(f"{c['variable']:<15} | {c['correlation']:>+12.4f} | {c['p_value']:>10.4f} | {sig:>10}")

    # Analyze by direction
    print("\n" + "=" * 70)
    print("CORRELATION BY DIRECTION")
    print("=" * 70)

    for direction in ['long', 'short']:
        sub = df[df['direction'] == direction]
        print(f"\n{direction.upper()} trades ({len(sub)} total):")

        for var in ['z_score_abs', 'hurst', 'volatility', 'beta']:
            if var not in sub.columns:
                continue
            valid = sub[[var, 'win']].dropna()
            if len(valid) < 5:
                continue
            corr, pval = stats.pointbiserialr(valid['win'], valid[var])
            sig = "*" if pval < 0.05 else ""
            print(f"  {var:<15}: corr={corr:+.4f} (p={pval:.4f}) {sig}")

    # Analyze by quintiles
    print("\n" + "=" * 70)
    print("HIT RATE BY QUINTILE")
    print("=" * 70)

    best_var = correlations[0]['variable'] if correlations else 'z_score_abs'

    for var in ['z_score_abs', 'hurst', 'volatility']:
        print(f"\n{var}:")
        try:
            df['quintile'] = pd.qcut(df[var], 5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'], duplicates='drop')
        except ValueError:
            # Not enough unique values for 5 quintiles
            try:
                n_bins = df[var].nunique()
                if n_bins < 3:
                    print(f"  Not enough unique values for quintile analysis")
                    continue
                df['quintile'] = pd.qcut(df[var], min(n_bins, 3), duplicates='drop')
            except:
                print(f"  Could not create quintiles")
                continue

        quintile_stats = df.groupby('quintile').agg({
            'win': ['mean', 'count'],
            'beta_adj_return': 'mean',
            var: 'mean'
        }).round(4)

        print(f"  {'Quintile':<12} | {'Hit Rate':>10} | {'Avg Beta%':>12} | {'Count':>6} | {var:>12}")
        print("  " + "-" * 60)
        for idx in quintile_stats.index:
            hr = quintile_stats.loc[idx, ('win', 'mean')] * 100
            cnt = int(quintile_stats.loc[idx, ('win', 'count')])
            ret = quintile_stats.loc[idx, ('beta_adj_return', 'mean')] * 100
            avg_var = quintile_stats.loc[idx, (var, 'mean')]
            print(f"  {str(idx):<12} | {hr:>9.1f}% | {ret:>+11.2f}% | {cnt:>6} | {avg_var:>12.3f}")

    # Best predictor analysis
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if correlations:
        best = correlations[0]
        print(f"\nBest predictor of win: {best['variable']}")
        print(f"  Correlation: {best['correlation']:+.4f}")
        print(f"  P-value: {best['p_value']:.4f}")

        if best['correlation'] > 0:
            print(f"\n  Interpretation: HIGHER {best['variable']} = MORE LIKELY TO WIN")
            print(f"  → Weight trades proportionally to {best['variable']}")
        else:
            print(f"\n  Interpretation: LOWER {best['variable']} = MORE LIKELY TO WIN")
            print(f"  → Weight trades inversely to {best['variable']}")

    # Propose weighting formula
    print("\n" + "=" * 70)
    print("PROPOSED WEIGHTING FORMULAS")
    print("=" * 70)

    for c in correlations[:3]:
        var = c['variable']
        corr = c['correlation']

        # Get range for normalization
        var_min = df[var].min()
        var_max = df[var].max()

        if corr > 0:
            print(f"\n{var} (positive correlation {corr:+.3f}):")
            print(f"  weight = 0.5 + ({var} - {var_min:.2f}) / ({var_max:.2f} - {var_min:.2f}) * 1.5")
            print(f"  → Scales weight from 0.5x to 2.0x based on {var}")
        else:
            print(f"\n{var} (negative correlation {corr:+.3f}):")
            print(f"  weight = 2.0 - ({var} - {var_min:.2f}) / ({var_max:.2f} - {var_min:.2f}) * 1.5")
            print(f"  → Scales weight from 2.0x to 0.5x based on {var}")


if __name__ == "__main__":
    main()
