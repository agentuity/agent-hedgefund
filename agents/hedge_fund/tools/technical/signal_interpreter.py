"""
Signal Interpreter Module

Converts raw technical indicator calculations into actionable trading signals.
Each function takes raw data and returns signal strength, direction, and explanation.
"""

from typing import Dict, List, Any, Optional
from enum import Enum

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class SignalDirection(Enum):
    """Signal direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

def interpret_sma_signal(prices: List[float], sma_values: List[float], period: int) -> Dict[str, Any]:
    """
    Interpret SMA signals for trend direction and strength
    """
    if not prices or not sma_values or len(prices) < 2:
        return {
            'direction': SignalDirection.NEUTRAL,
            'strength': SignalStrength.WEAK,
            'confidence': 0.0,
            'explanation': "Insufficient data for SMA analysis"
        }
    
    current_price = prices[-1]
    current_sma = sma_values[-1]
    
    # Price position relative to SMA
    price_above_sma = current_price > current_sma
    distance_percentage = abs(current_price - current_sma) / current_sma * 100
    
    # Direction
    if price_above_sma:
        direction = SignalDirection.BULLISH
        explanation = f"Price (${current_price:.2f}) is {distance_percentage:.1f}% above SMA{period} (${current_sma:.2f})"
    else:
        direction = SignalDirection.BEARISH
        explanation = f"Price (${current_price:.2f}) is {distance_percentage:.1f}% below SMA{period} (${current_sma:.2f})"
    
    if distance_percentage < 0.5:
        strength = SignalStrength.WEAK
        confidence = 0.4
    elif distance_percentage < 2:
        strength = SignalStrength.MODERATE
        confidence = 0.65
    elif distance_percentage < 4:
        strength = SignalStrength.STRONG
        confidence = 0.8
    else:
        strength = SignalStrength.VERY_STRONG
        confidence = 0.9
    
    return {
        'direction': direction,
        'strength': strength,
        'confidence': confidence,
        'explanation': explanation,
        'distance_percentage': distance_percentage,
        'price_above_sma': price_above_sma
    }

def interpret_ema_signal(prices: List[float], 
                        ema_fast: List[float], 
                        ema_slow: List[float],
                        fast_period: int = 8,
                        slow_period: int = 21) -> Dict[str, Any]:
    """
    Interpret EMA crossover signals (commonly EMA 8/21)
    """
    if not prices or not ema_fast or not ema_slow or len(ema_fast) < 2 or len(ema_slow) < 2:
        return {
            'direction': SignalDirection.NEUTRAL,
            'strength': SignalStrength.WEAK,
            'confidence': 0.0,
            'explanation': "Insufficient data for EMA crossover analysis"
        }
    
    current_price = prices[-1]
    current_fast = ema_fast[-1]
    current_slow = ema_slow[-1]
    prev_fast = ema_fast[-2]
    prev_slow = ema_slow[-2]
    
    # Current position
    fast_above_slow = current_fast > current_slow
    crossover_distance = abs(current_fast - current_slow) / current_slow * 100
    
    # Check for crossover
    bullish_crossover = prev_fast <= prev_slow and current_fast > current_slow
    bearish_crossover = prev_fast >= prev_slow and current_fast < current_slow
    
    # Price position relative to EMAs
    price_above_both = current_price > current_fast and current_price > current_slow
    price_below_both = current_price < current_fast and current_price < current_slow
    
    # Direction and explanation
    if bullish_crossover:
        direction = SignalDirection.BULLISH
        explanation = f"BULLISH: EMA{fast_period} crossed above EMA{slow_period}. Distance: {crossover_distance:.1f}%"
        confidence = min(0.9, 0.5 + crossover_distance * 0.1)
    elif bearish_crossover:
        direction = SignalDirection.BEARISH
        explanation = f"BEARISH: EMA{fast_period} crossed below EMA{slow_period}. Distance: {crossover_distance:.1f}%"
        confidence = min(0.9, 0.5 + crossover_distance * 0.1)
    elif fast_above_slow:
        direction = SignalDirection.BULLISH
        explanation = f"EMA{fast_period} above EMA{slow_period} by {crossover_distance:.1f}%"
        confidence = min(0.75, 0.4 + crossover_distance * 0.08)
    else:
        direction = SignalDirection.BEARISH
        explanation = f"EMA{fast_period} below EMA{slow_period} by {crossover_distance:.1f}%"
        confidence = min(0.75, 0.4 + crossover_distance * 0.08)
    
    # Strength calculation
    if bullish_crossover or bearish_crossover:
        if crossover_distance > 2:
            strength = SignalStrength.VERY_STRONG
        elif crossover_distance > 1:
            strength = SignalStrength.STRONG
        else:
            strength = SignalStrength.MODERATE
    else:
        if crossover_distance > 2.5:
            strength = SignalStrength.STRONG
        elif crossover_distance > 1.0:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
    
    return {
        'direction': direction,
        'strength': strength,
        'confidence': confidence,
        'explanation': explanation,
        'crossover_distance': crossover_distance,
        'bullish_crossover': bullish_crossover,
        'bearish_crossover': bearish_crossover,
        'price_above_both': price_above_both,
        'price_below_both': price_below_both
    }

def interpret_rsi_signal(rsi_values: List[float]) -> Dict[str, Any]:
    """
    Interpret RSI signals for overbought/oversold conditions
    """
    if not rsi_values:
        return {
            'direction': SignalDirection.NEUTRAL,
            'strength': SignalStrength.WEAK,
            'confidence': 0.0,
            'explanation': "No RSI data available"
        }
    
    current_rsi = rsi_values[-1]
    
    if current_rsi >= 70:
        direction = SignalDirection.BEARISH
        strength = SignalStrength.STRONG if current_rsi >= 80 else SignalStrength.MODERATE
        explanation = f"OVERBOUGHT: RSI {current_rsi:.1f} indicates potential selling pressure"
        confidence = min(0.9, (current_rsi - 70) / 30 * 0.5 + 0.5)
    elif current_rsi <= 30:
        direction = SignalDirection.BULLISH
        strength = SignalStrength.STRONG if current_rsi <= 20 else SignalStrength.MODERATE
        explanation = f"OVERSOLD: RSI {current_rsi:.1f} indicates potential buying opportunity"
        confidence = min(0.9, (30 - current_rsi) / 30 * 0.5 + 0.5)
    elif current_rsi < 40:
        direction = SignalDirection.BEARISH
        strength = SignalStrength.WEAK
        explanation = f"RSI {current_rsi:.1f} in bearish territory"
        confidence = 0.45
    elif current_rsi > 60:
        direction = SignalDirection.BULLISH
        strength = SignalStrength.WEAK
        explanation = f"RSI {current_rsi:.1f} in bullish territory"
        confidence = 0.45
    else:
        direction = SignalDirection.NEUTRAL
        strength = SignalStrength.WEAK
        explanation = f"RSI {current_rsi:.1f} in neutral range (40-60)"
        confidence = 0.25
    
    return {
        'direction': direction,
        'strength': strength,
        'confidence': confidence,
        'explanation': explanation,
        'current_rsi': current_rsi,
        'overbought': current_rsi >= 70,
        'oversold': current_rsi <= 30
    }

def interpret_macd_signal(macd_data: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Interpret MACD signals for momentum and trend changes
    """
    if not macd_data or 'macd_line' not in macd_data or 'signal_line' not in macd_data:
        return {
            'direction': SignalDirection.NEUTRAL,
            'strength': SignalStrength.WEAK,
            'confidence': 0.0,
            'explanation': "Insufficient MACD data"
        }
    
    macd_line = macd_data['macd_line']
    signal_line = macd_data['signal_line']
    histogram = macd_data.get('histogram', [])
    
    if len(macd_line) < 2 or len(signal_line) < 2:
        return {
            'direction': SignalDirection.NEUTRAL,
            'strength': SignalStrength.WEAK,
            'confidence': 0.0,
            'explanation': "Insufficient MACD data points"
        }
    
    current_macd = macd_line[-1]
    current_signal = signal_line[-1]
    prev_macd = macd_line[-2]
    prev_signal = signal_line[-2]
    
    # MACD position relative to signal line
    macd_above_signal = current_macd > current_signal
    distance = abs(current_macd - current_signal)
    
    # Check for crossovers
    bullish_crossover = prev_macd <= prev_signal and current_macd > current_signal
    bearish_crossover = prev_macd >= prev_signal and current_macd < current_signal
    
    # MACD zero line position
    macd_above_zero = current_macd > 0
    
    # Direction and explanation
    if bullish_crossover:
        direction = SignalDirection.BULLISH
        strength = SignalStrength.STRONG
        explanation = f"BULLISH: MACD crossed above signal line. MACD: {current_macd:.4f}, Signal: {current_signal:.4f}"
        confidence = 0.8
    elif bearish_crossover:
        direction = SignalDirection.BEARISH
        strength = SignalStrength.STRONG
        explanation = f"BEARISH: MACD crossed below signal line. MACD: {current_macd:.4f}, Signal: {current_signal:.4f}"
        confidence = 0.8
    elif macd_above_signal and macd_above_zero:
        direction = SignalDirection.BULLISH
        strength = SignalStrength.MODERATE
        explanation = f"MACD above signal line and zero line. Strong bullish momentum"
        confidence = 0.6
    elif not macd_above_signal and not macd_above_zero:
        direction = SignalDirection.BEARISH
        strength = SignalStrength.MODERATE
        explanation = f"MACD below signal line and zero line. Strong bearish momentum"
        confidence = 0.6
    elif macd_above_signal:
        direction = SignalDirection.BULLISH
        strength = SignalStrength.WEAK
        explanation = f"MACD above signal line but below zero. Weak bullish signal"
        confidence = 0.4
    else:
        direction = SignalDirection.BEARISH
        strength = SignalStrength.WEAK
        explanation = f"MACD below signal line but above zero. Weak bearish signal"
        confidence = 0.4
    
    return {
        'direction': direction,
        'strength': strength,
        'confidence': confidence,
        'explanation': explanation,
        'current_macd': current_macd,
        'current_signal': current_signal,
        'bullish_crossover': bullish_crossover,
        'bearish_crossover': bearish_crossover,
        'macd_above_signal': macd_above_signal,
        'macd_above_zero': macd_above_zero,
        'distance': distance
    }

def interpret_ema_crossover_signal(crossover_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interpret EMA crossover information into actionable signals
    """
    if not crossover_data:
        return {
            'direction': SignalDirection.NEUTRAL,
            'strength': SignalStrength.WEAK,
            'confidence': 0.0,
            'explanation': "No crossover data available"
        }
    
    current_position = crossover_data.get('current_position', 'neutral')
    recent_crossover = crossover_data.get('recent_crossover')
    crossover_strength = crossover_data.get('crossover_strength', 0.0)
    
    if recent_crossover == 'bullish':
        direction = SignalDirection.BULLISH
        explanation = f"FRESH BULLISH CROSSOVER detected! Strength: {crossover_strength:.3f}"
        confidence = min(0.9, 0.6 + crossover_strength * 0.3)
        strength = SignalStrength.VERY_STRONG if crossover_strength > 0.5 else SignalStrength.STRONG
    elif recent_crossover == 'bearish':
        direction = SignalDirection.BEARISH
        explanation = f"FRESH BEARISH CROSSOVER detected! Strength: {crossover_strength:.3f}"
        confidence = min(0.9, 0.6 + crossover_strength * 0.3)
        strength = SignalStrength.VERY_STRONG if crossover_strength > 0.5 else SignalStrength.STRONG
    elif current_position == 'above':
        direction = SignalDirection.BULLISH
        strength = SignalStrength.MODERATE
        explanation = "Fast EMA above slow EMA - bullish trend continuing"
        confidence = 0.5
    elif current_position == 'below':
        direction = SignalDirection.BEARISH
        strength = SignalStrength.MODERATE
        explanation = "Fast EMA below slow EMA - bearish trend continuing"
        confidence = 0.5
    else:
        direction = SignalDirection.NEUTRAL
        strength = SignalStrength.WEAK
        explanation = "EMAs are converging - trend unclear"
        confidence = 0.2
    
    return {
        'direction': direction,
        'strength': strength,
        'confidence': confidence,
        'explanation': explanation,
        'recent_crossover': recent_crossover,
        'crossover_strength': crossover_strength,
        'current_position': current_position
    } 