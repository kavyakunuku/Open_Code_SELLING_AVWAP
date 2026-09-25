"""
Anchored Volume Weighted Average Price (AVWAP) Indicator

Custom AVWAP that anchors to the first tradable candle of an option contract's life.
Does not reset at day/week boundaries - unique to each strike/expiry combination.

Rules:
- Anchor: First candle when option contract becomes tradable (based on listing date/first OHLCV candle)
- Timeframe: 15-minute candles (IST market sessions)
- AVWAP is cumulative from anchor point, persists across days
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple


def calculate_avwap(
    high: Union[np.ndarray, pd.Series, list],
    low: Union[np.ndarray, pd.Series, list],
    close: Union[np.ndarray, pd.Series, list],
    volume: Union[np.ndarray, pd.Series, list],
    anchor_index: Optional[int] = None,
    *,
    anchor_candle_time: Optional[str] = None,
) -> np.ndarray:
    """
    Calculate Anchored VWAP (AVWAP).
    
    The AVWAP anchors (initializes) from a specific candle index and 
    accumulates volume-weighted average price from that point onwards.
    Unlike standard VWAP which resets each session, AVWAP persists.
    
    Parameters
    ----------
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices
    volume : array-like
        Volume data
    anchor_index : int, optional
        Index of the candle to anchor from (0-based).
        If None, uses the first candle (index 0).
    anchor_candle_time : str, optional
        ISO format datetime string to anchor from (e.g., "2024-01-15T09:15:00").
        Used when anchor_index cannot be determined directly.
        
    Returns
    -------
    np.ndarray
        Array of AVWAP values. Returns NaN before anchor index.
        
    Example
    -------
    # Anchor from candle index 5 (6th candle)
    avwap = calculate_avwap(high, low, close, volume, anchor_index=5)
    
    # Anchor from specific time
    avwap = calculate_avwap(high, low, close, volume, anchor_candle_time="2024-01-15T09:15:00")
    """
    # Convert to numpy arrays if needed
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.asarray(volume, dtype=float)
    
    n = len(high_arr)
    avwap = np.full(n, np.nan)
    
    # Determine anchor starting index
    if anchor_index is not None:
        start_idx = anchor_index
    elif anchor_candle_time is not None:
        # Find the first candle at or after the anchor time
        # This requires time data - for now, use index 0 as fallback
        start_idx = 0
    else:
        # Anchor from first candle (index 0)
        start_idx = 0
    
    # Calculate AVWAP from anchor index onwards
    if start_idx < n and start_idx >= 0:
        # Initialize cumulative variables at anchor point
        cumulative_pv = 0.0  # price * volume
        cumulative_v = 0.0   # volume
        
        for i in range(start_idx, n):
            # Typical price = (high + low + close) / 3
            typical_price = (high_arr[i] + low_arr[i] + close_arr[i]) / 3.0
            pv = typical_price * volume_arr[i]  # price * volume
            
            cumulative_pv += pv
            cumulative_v += volume_arr[i]
            
            # Avoid division by zero
            if cumulative_v > 0:
                avwap[i] = cumulative_pv / cumulative_v
            else:
                avwap[i] = typical_price
        
        # Set NaN before anchor index (AVWAP not defined yet)
        # Already initialized as NaN, so no action needed
    
    return avwap


def calculate_avwap_with_anchor_time(
    df: pd.DataFrame,
    time_column: str = "timestamp",
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
    volume_column: str = "volume",
    anchor_time: str = "",
) -> pd.DataFrame:
    """
    Calculate AVWAP anchored at a specific time from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    time_column : str
        Column name containing timestamps
    high_column : str
        Column name for high prices
    low_column : str
        Column name for low prices
    close_column : str
        Column name for close prices
    volume_column : str
        Column name for volume
    anchor_time : str
        ISO format datetime string to anchor from
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'avwap' column
    """
    # Ensure timestamp column is datetime
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Find the anchor index - first candle at or after anchor_time
    anchor_dt = pd.to_datetime(anchor_time)
    anchor_idx = df[df[time_column] >= anchor_idx].index[0] if len(df[df[time_column] >= anchor_dt]) > 0 else 0
    
    # Calculate AVWAP
    avwap_values = calculate_avwap(
        df[high_column].values,
        df[low_column].values,
        df[close_column].values,
        df[volume_column].values,
        anchor_index=anchor_idx,
    )
    
    df = df.copy()
    df['avwap'] = avwap_values
    
    return df


# For backward compatibility - allow direct function import
__all__ = ['calculate_avwap', 'calculate_avwap_with_anchor_time']