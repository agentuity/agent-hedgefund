"""
Technical Indicators Module

Pure mathematical functions for calculating technical indicators.
No market data fetching or signal interpretation - just calculations.
"""

from typing import List, Dict, Any
import numpy as np

def calculate_sma(prices: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average"""
    if not prices or len(prices) < period:
        raise ValueError(f"Not enough data for SMA period {period}. Need {period}, got {len(prices)}.")
    return [float(np.mean(prices[i-period+1:i+1])) for i in range(period-1, len(prices))]

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if not prices or len(prices) < period:
        raise ValueError(f"Not enough data for EMA period {period}. Need {period}, got {len(prices)}.")
    
    ema_values = []
    k = 2 / (period + 1)
    # Initial EMA is SMA of first period
    ema_prev = float(np.mean(prices[:period]))
    ema_values.append(ema_prev)
    
    # Calculate for subsequent prices
    for price in prices[period:]:
        ema_curr = price * k + ema_prev * (1 - k)
        ema_values.append(ema_curr)
        ema_prev = ema_curr
    
    return ema_values

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index"""
    if not prices or len(prices) < period + 1:
        raise ValueError(f"Not enough data for RSI period {period}. Need {period+1}, got {len(prices)}.")
    
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        rs = np.inf
    else:
        rs = up / down
    
    rsi_values = [100.0 - (100.0 / (1.0 + rs))]
    
    for i in range(period, len(deltas)):
        delta = deltas[i]
        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        if down == 0:
            rs = np.inf
        else:
            rs = up / down
        
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    
    return rsi_values

def calculate_macd(prices: List[float], 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> Dict[str, List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns dict with macd_line, signal_line, and histogram
    """
    if not prices or len(prices) < slow_period:
        raise ValueError(f"Not enough data for MACD. Need {slow_period}, got {len(prices)}.")
    
    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)
    
    min_length = min(len(fast_ema), len(slow_ema))
    aligned_fast_ema = fast_ema[-min_length:]
    aligned_slow_ema = slow_ema[-min_length:]
    
    # Calculate MACD line
    macd_line = [fast - slow for fast, slow in zip(aligned_fast_ema, aligned_slow_ema)]
    
    # Calculate signal line
    if len(macd_line) < signal_period:
        raise ValueError(f"Not enough MACD data for signal line. Need {signal_period}, got {len(macd_line)}.")
    
    signal_line = calculate_ema(macd_line, signal_period)
    
    # Calculate histogram - align to signal line length
    start_index_hist = len(macd_line) - len(signal_line)
    aligned_macd_line = macd_line[start_index_hist:]
    histogram = [macd - signal for macd, signal in zip(aligned_macd_line, signal_line)]
    
    return {
        'macd_line': aligned_macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }

def detect_ema_crossover(ema_fast: List[float], ema_slow: List[float]) -> Dict[str, Any]:
    """
    Detect EMA crossovers and return crossover information
    """
    if len(ema_fast) < 2 or len(ema_slow) < 2:
        return {
            'current_position': 'neutral',
            'recent_crossover': None,
            'crossover_strength': 0.0
        }
    
    # Align EMAs to same length
    min_length = min(len(ema_fast), len(ema_slow))
    fast_aligned = ema_fast[-min_length:]
    slow_aligned = ema_slow[-min_length:]
    
    # Current position
    current_fast = fast_aligned[-1]
    current_slow = slow_aligned[-1]
    
    if current_fast > current_slow:
        current_position = 'above'
    elif current_fast < current_slow:
        current_position = 'below'
    else:
        current_position = 'neutral'
    
    recent_crossover = None
    crossover_strength = (
                abs(current_fast - current_slow) / current_slow * 100
                if current_slow != 0
                else 0.0
            )
    
    if len(fast_aligned) >= 3:
        prev_fast = fast_aligned[-2]
        prev_slow = slow_aligned[-2]
        
        if prev_fast <= prev_slow and current_fast > current_slow:
            recent_crossover = 'bullish'
        elif prev_fast >= prev_slow and current_fast < current_slow:
            recent_crossover = 'bearish'
    
    return {
        'current_position': current_position,
        'recent_crossover': recent_crossover,
        'crossover_strength': crossover_strength
    }

def calculate_volume_indicators(volumes: List[float], 
                              prices: List[float], 
                              period: int = 20) -> Dict[str, Any]:
    """
    Calculate volume-based indicators
    Returns volume ratios and trend analysis
    """
    if len(volumes) < period or len(prices) < period:
        raise ValueError(f"Not enough data for volume indicators. Need {period} periods.")
    
    # Volume SMA
    volume_sma = calculate_sma(volumes, period)
    
    # Volume ratios
    volume_ratios = []
    for i in range(len(volume_sma)):
        current_volume = volumes[i + period - 1]
        avg_volume = volume_sma[i]
        if avg_volume > 0:
            volume_ratios.append(current_volume / avg_volume)
        else:
            volume_ratios.append(1.0)
    
    # Volume trend analysis
    recent_volumes = volumes[-min(5, len(volumes)):]
    recent_prices = prices[-min(5, len(prices)):]
    
    volume_changes = [recent_volumes[i] - recent_volumes[i-1] for i in range(1, len(recent_volumes))]
    price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
    
    volume_trend = "increasing" if sum(volume_changes) > 0 else "decreasing"
    price_trend = "increasing" if sum(price_changes) > 0 else "decreasing"
    
    # Price-volume relationship
    if volume_trend == "increasing" and price_trend == "increasing":
        relationship = "bullish_confirmation"
    elif volume_trend == "increasing" and price_trend == "decreasing":
        relationship = "bearish_confirmation"
    elif volume_trend == "decreasing" and price_trend == "increasing":
        relationship = "weak_bullish"
    elif volume_trend == "decreasing" and price_trend == "decreasing":
        relationship = "weak_bearish"
    else:
        relationship = "neutral"
    
    return {
        'volume_ratios': volume_ratios,
        'volume_trend': volume_trend,
        'price_trend': price_trend,
        'relationship': relationship,
        'avg_volume_change': np.mean(volume_changes) if volume_changes else 0.0,
        'avg_price_change': np.mean(price_changes) if price_changes else 0.0
    } 