"""
Extreme Hurst Trading Signal - Pure Hurst with Dynamic Inertia
Replication of Parallax's extreme hurst technical signal

Dynamic entry: Threshold adapts based on Hurst (mean-reversion strength)
Dynamic exit: Hold period based on extension inertia and Hurst
No RSI - pure Hurst-based model
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


def calculate_hurst(prices: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate rolling Hurst exponent using variance ratio method

    H < 0.5: Mean-reverting (lower = stronger mean reversion)
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    n = len(prices)
    hurst = np.full(n, 0.5)
    log_prices = np.log(np.maximum(prices, 1e-10))

    for i in range(lookback, n, 5):  # Every 5 bars
        segment = log_prices[max(0, i-lookback):i]
        if len(segment) >= 15:
            lags = [2, 4, 8]
            taus = []
            for lag in lags:
                if lag < len(segment):
                    diff = segment[lag:] - segment[:-lag]
                    std_val = np.std(diff)
                    if std_val > 0:
                        taus.append((lag, std_val))
            if len(taus) >= 2:
                log_lags = np.log([t[0] for t in taus])
                log_taus = np.log([t[1] for t in taus])
                slope, _, _, _, _ = stats.linregress(log_lags, log_taus)
                hurst[i] = np.clip(slope, 0.0, 1.0)

    # Forward fill
    for i in range(1, n):
        if hurst[i] == 0.5 and hurst[i-1] != 0.5 and i % 5 != 0:
            hurst[i] = hurst[i-1]

    return hurst


def calculate_zscore(close: np.ndarray, lookback: int) -> np.ndarray:
    """Calculate rolling z-score of price"""
    roll_mean = pd.Series(close).rolling(lookback).mean().values
    roll_std = pd.Series(close).rolling(lookback).std().values
    roll_std = np.where(roll_std > 0, roll_std, 1e-10)
    return (close - roll_mean) / roll_std


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range for volatility-based inertia"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


def calculate_formation_duration(close: np.ndarray, zscore: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """
    Calculate how long price has been in an extended state (formation duration)

    This measures the "duration of log formation" - how many bars the price has been
    continuously above/below a threshold. Longer formations = more accumulated energy
    = potentially longer time to revert.
    """
    n = len(close)
    duration = np.zeros(n)

    for i in range(1, n):
        # Count consecutive bars in extended state
        if abs(zscore[i]) > threshold:
            # Check if same direction as previous
            if zscore[i] * zscore[i-1] > 0:  # Same sign
                duration[i] = duration[i-1] + 1
            else:
                duration[i] = 1
        else:
            duration[i] = 0

    return duration


def calculate_wave_amplitude(close: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate the amplitude of recent price waves

    This measures the "amplitude of waves" - the magnitude of price swings.
    Larger amplitudes indicate more momentum that needs to dissipate.
    """
    n = len(close)
    amplitude = np.zeros(n)

    for i in range(lookback, n):
        segment = close[max(0, i-lookback):i+1]
        # Amplitude = (max - min) / mean, normalized
        range_val = segment.max() - segment.min()
        mean_val = segment.mean()
        if mean_val > 0:
            amplitude[i] = range_val / mean_val

    return amplitude


class ExtremeHurstSignal:
    """
    Pure Hurst-based signal with dynamic entry and exit based on inertia

    Inertia Concept:
    - Entry threshold adapts to Hurst: Lower Hurst = more mean-reverting = lower threshold
    - Exit bars adapt to extension + Hurst: Larger extension + lower Hurst = faster reversion expected

    The "inertia" represents how strongly price tends to continue vs revert:
    - Low inertia (low Hurst) = quick reversion, can enter smaller moves, exit faster
    - High inertia (high Hurst) = slow reversion, need larger moves, hold longer
    """

    def __init__(
        self,
        base_threshold: float = 2.5,
        extension_lookback: int = 15,
        hurst_lookback: int = 20,
        max_hurst: float = 0.55,
        min_exit_bars: int = 3,
        max_exit_bars: int = 15,
        inertia_sensitivity: float = 1.0,
    ):
        """
        Initialize with dynamic parameters

        Args:
            base_threshold: Base z-score threshold (adjusted by Hurst)
            extension_lookback: Lookback for z-score
            hurst_lookback: Lookback for Hurst calculation
            max_hurst: Maximum Hurst to generate signals
            min_exit_bars: Minimum hold period
            max_exit_bars: Maximum hold period
            inertia_sensitivity: How much Hurst affects dynamics (1.0 = normal)
        """
        self.base_threshold = base_threshold
        self.extension_lookback = extension_lookback
        self.hurst_lookback = hurst_lookback
        self.max_hurst = max_hurst
        self.min_exit_bars = min_exit_bars
        self.max_exit_bars = max_exit_bars
        self.inertia_sensitivity = inertia_sensitivity

    def get_dynamic_threshold(self, hurst: float) -> float:
        """
        Dynamic entry threshold based on Hurst (inertia)

        Lower Hurst = more mean-reverting = can enter at smaller extensions
        Higher Hurst = less mean-reverting = need larger extensions
        """
        # Scale threshold: H=0.3 -> 0.7x base, H=0.5 -> 1.0x base, H=0.6 -> 1.2x base
        hurst_factor = 0.5 + hurst * self.inertia_sensitivity
        return self.base_threshold * hurst_factor

    def get_dynamic_exit_bars(self, hurst: float, zscore: float) -> int:
        """
        Dynamic exit bars based on inertia

        - Lower Hurst = faster mean reversion = shorter hold
        - Larger extension = may need more time = longer hold
        - Balance between the two
        """
        # Hurst component: lower Hurst = shorter hold
        # H=0.3 -> 0.6x, H=0.5 -> 1.0x, H=0.6 -> 1.2x
        hurst_factor = 0.5 + hurst * self.inertia_sensitivity

        # Extension component: larger extension = longer hold (takes more time to revert)
        # z=2 -> 0.8x, z=3 -> 1.0x, z=4 -> 1.2x
        extension_factor = 0.6 + abs(zscore) * 0.15

        # Combine: base * hurst_factor * extension_factor
        base_bars = (self.min_exit_bars + self.max_exit_bars) / 2
        dynamic_bars = base_bars * hurst_factor * extension_factor

        return int(np.clip(dynamic_bars, self.min_exit_bars, self.max_exit_bars))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with dynamic entry and exit"""
        df = df.copy()
        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float) if 'High' in df.columns else close
        low = df['Low'].values.astype(float) if 'Low' in df.columns else close
        n = len(close)

        # Calculate indicators
        hurst = calculate_hurst(close, self.hurst_lookback)
        zscore = calculate_zscore(close, self.extension_lookback)
        atr = calculate_atr(high, low, close)

        # Normalize ATR for volatility-adjusted inertia
        atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().values
        atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)

        signals = np.zeros(n)
        position_sizes = np.zeros(n)
        exit_bars_arr = np.zeros(n, dtype=int)
        dynamic_thresholds = np.zeros(n)

        min_idx = max(self.hurst_lookback, self.extension_lookback, 20)

        for i in range(min_idx, n):
            h = hurst[i]
            z = zscore[i]
            vol_adj = atr_ratio[i]

            # Dynamic threshold adjusted for Hurst and volatility
            threshold = self.get_dynamic_threshold(h) * (0.8 + vol_adj * 0.4)
            dynamic_thresholds[i] = threshold

            # Check if Hurst indicates mean-reverting regime
            if h >= self.max_hurst:
                continue

            # Long signal: Bottom extension
            if z < -threshold:
                signals[i] = 1
                exit_bars = self.get_dynamic_exit_bars(h, z)
                exit_bars_arr[i] = exit_bars

                # Position size based on signal strength (inertia-adjusted)
                extension_strength = min(abs(z) / threshold, 2.0)
                hurst_boost = 1.0 + max(0, 0.5 - h) * 2 * self.inertia_sensitivity
                position_sizes[i] = min(extension_strength * hurst_boost, 3.0)

            # Short signal: Top extension
            elif z > threshold:
                signals[i] = -1
                exit_bars = self.get_dynamic_exit_bars(h, z)
                exit_bars_arr[i] = exit_bars

                extension_strength = min(abs(z) / threshold, 2.0)
                hurst_boost = 1.0 + max(0, 0.5 - h) * 2 * self.inertia_sensitivity
                position_sizes[i] = min(extension_strength * hurst_boost, 3.0)

        df['signal'] = signals
        df['position_size'] = position_sizes
        df['exit_bars'] = exit_bars_arr
        df['dynamic_threshold'] = dynamic_thresholds
        df['hurst'] = hurst
        df['zscore'] = zscore
        df['atr_ratio'] = atr_ratio

        return df

    def backtest(self, df: pd.DataFrame) -> Dict:
        """Backtest with dynamic exit bars per trade"""
        df = self.generate_signals(df)
        close = df['Close'].values
        signals = df['signal'].values
        position_sizes = df['position_size'].values
        exit_bars_arr = df['exit_bars'].values

        trades = []
        n = len(df)

        for i in range(n):
            if signals[i] != 0:
                exit_bars = int(exit_bars_arr[i])
                if i + exit_bars >= n:
                    continue

                entry_price = close[i]
                exit_price = close[i + exit_bars]
                direction = signals[i]
                size = position_sizes[i]

                if direction == 1:
                    pct_return = (exit_price - entry_price) / entry_price
                else:
                    pct_return = (entry_price - exit_price) / entry_price

                trades.append({
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_bars': exit_bars,
                    'direction': direction,
                    'position_size': size,
                    'pct_return': pct_return,
                    'sized_return': pct_return * size,
                    'win': pct_return > 0
                })

        if not trades:
            return {'total_trades': 0, 'long_trades': 0, 'short_trades': 0,
                    'long_hit_rate': 0, 'short_hit_rate': 0}

        trades_df = pd.DataFrame(trades)
        long_trades = trades_df[trades_df['direction'] == 1]
        short_trades = trades_df[trades_df['direction'] == -1]

        return {
            'total_trades': len(trades_df),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_wins': long_trades['win'].sum() if len(long_trades) > 0 else 0,
            'short_wins': short_trades['win'].sum() if len(short_trades) > 0 else 0,
            'long_hit_rate': long_trades['win'].mean() * 100 if len(long_trades) > 0 else 0,
            'short_hit_rate': short_trades['win'].mean() * 100 if len(short_trades) > 0 else 0,
            'overall_hit_rate': trades_df['win'].mean() * 100,
            'avg_return': trades_df['pct_return'].mean() * 100,
            'avg_sized_return': trades_df['sized_return'].mean() * 100,
            'avg_exit_bars': trades_df['exit_bars'].mean(),
            'trades': trades_df
        }
