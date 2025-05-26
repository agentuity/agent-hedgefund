from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import numpy as np
import yfinance as yf
from pycoingecko import CoinGeckoAPI

# Placeholder for actual data fetching libraries
# For stocks: import yfinance as yf
# For crypto: from pycoingecko import CoinGeckoAPI (or import requests)

# --- Pydantic Schemas ---

class AssetType(str, Enum):
    STOCK = "STOCK"
    CRYPTO = "CRYPTO"

class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"

class TechnicalSignal(BaseModel):
    signal: SignalType
    strength: float = Field(..., description="Signal strength from 0.0 to 1.0")
    reason: str = Field(..., description="Human-readable explanation of the signal")

class IndicatorSpec(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = None

class TechnicalAnalysisRequest(BaseModel):
    asset_type: AssetType
    symbol: str
    timeframe: str = Field(..., description="e.g., '1d', '1h', '5m' compatible with chosen APIs")
    indicators: List[IndicatorSpec]

class IndicatorResult(BaseModel):
    name: str
    values: Optional[Any] = None
    error: Optional[str] = None
    current_value: Optional[float] = None  # Most recent value
    signal: Optional[TechnicalSignal] = None  # Actionable interpretation
    
class TechnicalAnalysisToolOutput(BaseModel):
    asset_type: AssetType
    symbol: str
    timeframe: str
    current_price: Optional[float] = None  # Most recent price
    indicators: List[IndicatorResult]
    data_fetch_error: Optional[str] = None

# --- Indicator Functions ---

def sma(prices: List[float], period: int) -> List[float]:
    if not prices or len(prices) < period:
        raise ValueError(f"Not enough data for SMA period {period}. Need {period}, got {len(prices)}.")
    return [float(np.mean(prices[i-period+1:i+1])) for i in range(period-1, len(prices))]

def ema(prices: List[float], period: int) -> List[float]:
    if not prices or len(prices) < period:
        raise ValueError(f"Not enough data for EMA period {period}. Need {period}, got {len(prices)}.")
    ema_values = []
    k = 2 / (period + 1)
    # Initial EMA can be SMA or just the first 'period' average
    ema_prev = float(np.mean(prices[:period]))
    ema_values.append(ema_prev) # Add EMA for the first period window
    # Calculate for subsequent prices
    for price in prices[period:]:
        ema_curr = price * k + ema_prev * (1 - k)
        ema_values.append(ema_curr)
        ema_prev = ema_curr
    # The returned list will be shorter than prices by (period -1)
    # To align with price length, often first period-1 values are NaN or not returned.
    # For simplicity, returning calculated values.
    return ema_values

def rsi(prices: List[float], period: int = 14) -> List[float]:
    if not prices or len(prices) < period + 1: # RSI needs at least period+1 prices for initial calculation
        raise ValueError(f"Not enough data for RSI period {period}. Need {period+1}, got {len(prices)}.")
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        rs = np.inf # Or handle as per specific strategy, e.g., RSI = 100
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

def macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns: {
        'macd_line': MACD line (fast EMA - slow EMA),
        'signal_line': Signal line (EMA of MACD line),
        'histogram': Histogram (MACD line - signal line)
    }
    """
    if not prices or len(prices) < slow_period:
        raise ValueError(f"Not enough data for MACD. Need {slow_period}, got {len(prices)}.")
    
    # Calculate fast and slow EMAs
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)
    
    # Align the EMAs (slow EMA is shorter, so we need to match lengths)
    # The slow EMA starts later, so we trim the fast EMA to match
    start_index = slow_period - fast_period
    aligned_fast_ema = fast_ema[start_index:]
    
    # Calculate MACD line (fast EMA - slow EMA)
    macd_line = [fast - slow for fast, slow in zip(aligned_fast_ema, slow_ema)]
    
    # Calculate signal line (EMA of MACD line)
    if len(macd_line) < signal_period:
        raise ValueError(f"Not enough MACD data for signal line. Need {signal_period}, got {len(macd_line)}.")
    
    signal_line = ema(macd_line, signal_period)
    
    # Calculate histogram (MACD line - signal line)
    # Align MACD line with signal line
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
    Returns: {
        'current_position': 'above'/'below'/'neutral',
        'recent_crossover': 'bullish'/'bearish'/None,
        'crossover_strength': float (0.0-1.0)
    }
    """
    if len(ema_fast) < 2 or len(ema_slow) < 2:
        return {
            'current_position': 'neutral',
            'recent_crossover': None,
            'crossover_strength': 0.0
        }
    
    # Align the EMAs to same length
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
    
    # Check for recent crossover (last few periods)
    recent_crossover = None
    crossover_strength = 0.0
    
    if len(fast_aligned) >= 3:
        # Look at last 3 periods for crossover
        prev_fast = fast_aligned[-2]
        prev_slow = slow_aligned[-2]
        
        # Bullish crossover: fast was below, now above
        if prev_fast <= prev_slow and current_fast > current_slow:
            recent_crossover = 'bullish'
            # Strength based on how much the fast EMA is above slow EMA
            crossover_strength = min(abs(current_fast - current_slow) / current_slow * 100, 1.0)
        
        # Bearish crossover: fast was above, now below
        elif prev_fast >= prev_slow and current_fast < current_slow:
            recent_crossover = 'bearish'
            crossover_strength = min(abs(current_fast - current_slow) / current_slow * 100, 1.0)
    
    return {
        'current_position': current_position,
        'recent_crossover': recent_crossover,
        'crossover_strength': crossover_strength
    }

def volume_sma(volumes: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average of volume"""
    if not volumes or len(volumes) < period:
        raise ValueError(f"Not enough volume data for SMA period {period}. Need {period}, got {len(volumes)}.")
    return [float(np.mean(volumes[i-period+1:i+1])) for i in range(period-1, len(volumes))]

def volume_ratio(volumes: List[float], period: int = 20) -> List[float]:
    """Calculate volume ratio (current volume / average volume)"""
    if not volumes or len(volumes) < period:
        raise ValueError(f"Not enough volume data for ratio period {period}. Need {period}, got {len(volumes)}.")
    
    volume_avg = volume_sma(volumes, period)
    ratios = []
    
    # Calculate ratios for the period where we have averages
    for i in range(len(volume_avg)):
        current_volume = volumes[i + period - 1]  # Align with SMA index
        avg_volume = volume_avg[i]
        if avg_volume > 0:
            ratios.append(current_volume / avg_volume)
        else:
            ratios.append(1.0)  # Neutral ratio if no average
    
    return ratios

def analyze_volume_trend(volumes: List[float], prices: List[float], period: int = 5) -> Dict[str, Any]:
    """
    Analyze volume trend and price-volume relationship
    Returns volume trend analysis
    """
    if len(volumes) < period or len(prices) < period:
        raise ValueError(f"Not enough data for volume trend analysis. Need {period} periods.")
    
    # Get recent volume and price data
    recent_volumes = volumes[-period:]
    recent_prices = prices[-period:]
    
    # Calculate volume trend (increasing/decreasing)
    volume_changes = [recent_volumes[i] - recent_volumes[i-1] for i in range(1, len(recent_volumes))]
    volume_trend = "increasing" if sum(volume_changes) > 0 else "decreasing"
    
    # Calculate price trend
    price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
    price_trend = "increasing" if sum(price_changes) > 0 else "decreasing"
    
    # Analyze price-volume relationship
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
        'volume_trend': volume_trend,
        'price_trend': price_trend,
        'relationship': relationship,
        'avg_volume_change': np.mean(volume_changes),
        'avg_price_change': np.mean(price_changes)
    }

# --- Signal Interpretation Functions ---

def interpret_sma_signal(prices: List[float], sma_values: List[float], period: int) -> TechnicalSignal:
    """Interpret SMA signal based on price vs SMA relationship"""
    current_price = prices[-1]
    current_sma = sma_values[-1]
    
    # Calculate trend strength based on how far price is from SMA
    price_diff_pct = ((current_price - current_sma) / current_sma) * 100
    
    if current_price > current_sma:
        if price_diff_pct > 2:  # Strong uptrend
            return TechnicalSignal(
                signal=SignalType.BUY,
                strength=min(abs(price_diff_pct) / 10, 1.0),  # Cap at 1.0
                reason=f"Price ${current_price:.2f} is {price_diff_pct:.1f}% above SMA({period}) ${current_sma:.2f} - Strong uptrend"
            )
        else:  # Mild uptrend
            return TechnicalSignal(
                signal=SignalType.HOLD,
                strength=0.5,
                reason=f"Price ${current_price:.2f} is {price_diff_pct:.1f}% above SMA({period}) ${current_sma:.2f} - Mild uptrend"
            )
    else:
        if price_diff_pct < -2:  # Strong downtrend
            return TechnicalSignal(
                signal=SignalType.SELL,
                strength=min(abs(price_diff_pct) / 10, 1.0),
                reason=f"Price ${current_price:.2f} is {abs(price_diff_pct):.1f}% below SMA({period}) ${current_sma:.2f} - Strong downtrend"
            )
        else:  # Mild downtrend
            return TechnicalSignal(
                signal=SignalType.HOLD,
                strength=0.3,
                reason=f"Price ${current_price:.2f} is {abs(price_diff_pct):.1f}% below SMA({period}) ${current_sma:.2f} - Mild downtrend"
            )

def interpret_ema_signal(prices: List[float], ema_values: List[float], period: int) -> TechnicalSignal:
    """Interpret EMA signal - similar to SMA but more responsive"""
    current_price = prices[-1]
    current_ema = ema_values[-1]
    
    # Look at recent trend (last 3 EMA values)
    if len(ema_values) >= 3:
        recent_ema_trend = (ema_values[-1] - ema_values[-3]) / ema_values[-3] * 100
    else:
        recent_ema_trend = 0
    
    price_diff_pct = ((current_price - current_ema) / current_ema) * 100
    
    if current_price > current_ema and recent_ema_trend > 0:
        return TechnicalSignal(
            signal=SignalType.BUY,
            strength=min((abs(price_diff_pct) + abs(recent_ema_trend)) / 15, 1.0),
            reason=f"Price ${current_price:.2f} above EMA({period}) ${current_ema:.2f} with rising EMA trend ({recent_ema_trend:.1f}%)"
        )
    elif current_price < current_ema and recent_ema_trend < 0:
        return TechnicalSignal(
            signal=SignalType.SELL,
            strength=min((abs(price_diff_pct) + abs(recent_ema_trend)) / 15, 1.0),
            reason=f"Price ${current_price:.2f} below EMA({period}) ${current_ema:.2f} with falling EMA trend ({recent_ema_trend:.1f}%)"
        )
    else:
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=0.4,
            reason=f"Price ${current_price:.2f} vs EMA({period}) ${current_ema:.2f} - Mixed signals"
        )

def interpret_rsi_signal(rsi_values: List[float], period: int) -> TechnicalSignal:
    """Interpret RSI signal based on overbought/oversold levels"""
    current_rsi = rsi_values[-1]
    
    if current_rsi > 70:
        strength = min((current_rsi - 70) / 30, 1.0)  # Scale from 70-100
        return TechnicalSignal(
            signal=SignalType.SELL,
            strength=strength,
            reason=f"RSI({period}) at {current_rsi:.1f} - Overbought territory (>70), potential sell signal"
        )
    elif current_rsi < 30:
        strength = min((30 - current_rsi) / 30, 1.0)  # Scale from 0-30
        return TechnicalSignal(
            signal=SignalType.BUY,
            strength=strength,
            reason=f"RSI({period}) at {current_rsi:.1f} - Oversold territory (<30), potential buy signal"
        )
    elif 45 <= current_rsi <= 55:
        return TechnicalSignal(
            signal=SignalType.NEUTRAL,
            strength=0.2,
            reason=f"RSI({period}) at {current_rsi:.1f} - Neutral zone, no clear signal"
        )
    else:
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=0.3,
            reason=f"RSI({period}) at {current_rsi:.1f} - Normal range, hold position"
        )

def interpret_macd_signal(macd_line: List[float], signal_line: List[float], histogram: List[float]) -> TechnicalSignal:
    """Interpret MACD signal based on line crossovers and histogram"""
    current_macd = macd_line[-1]
    current_signal = signal_line[-1]
    current_histogram = histogram[-1]
    
    # Check for recent crossover
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        
        # Bullish crossover: MACD crosses above signal line
        if prev_macd <= prev_signal and current_macd > current_signal:
            strength = min(abs(current_macd - current_signal) / abs(current_signal) * 10, 1.0)
            return TechnicalSignal(
                signal=SignalType.BUY,
                strength=strength,
                reason=f"MACD bullish crossover - MACD line {current_macd:.4f} crossed above signal line {current_signal:.4f}"
            )
        
        # Bearish crossover: MACD crosses below signal line
        elif prev_macd >= prev_signal and current_macd < current_signal:
            strength = min(abs(current_macd - current_signal) / abs(current_signal) * 10, 1.0)
            return TechnicalSignal(
                signal=SignalType.SELL,
                strength=strength,
                reason=f"MACD bearish crossover - MACD line {current_macd:.4f} crossed below signal line {current_signal:.4f}"
            )
    
    # No recent crossover, check current position and histogram trend
    if current_macd > current_signal and current_histogram > 0:
        # MACD above signal line with positive histogram
        histogram_strength = min(abs(current_histogram) / abs(current_macd) * 5, 0.7)
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=histogram_strength,
            reason=f"MACD above signal line with positive momentum (histogram: {current_histogram:.4f})"
        )
    elif current_macd < current_signal and current_histogram < 0:
        # MACD below signal line with negative histogram
        histogram_strength = min(abs(current_histogram) / abs(current_macd) * 5, 0.7)
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=histogram_strength,
            reason=f"MACD below signal line with negative momentum (histogram: {current_histogram:.4f})"
        )
    else:
        return TechnicalSignal(
            signal=SignalType.NEUTRAL,
            strength=0.2,
            reason=f"MACD mixed signals - MACD: {current_macd:.4f}, Signal: {current_signal:.4f}, Histogram: {current_histogram:.4f}"
        )

def interpret_ema_crossover_signal(ema_fast: List[float], ema_slow: List[float], fast_period: int, slow_period: int) -> TechnicalSignal:
    """Interpret EMA crossover signal for trend direction"""
    crossover_info = detect_ema_crossover(ema_fast, ema_slow)
    
    current_fast = ema_fast[-1]
    current_slow = ema_slow[-1]
    
    if crossover_info['recent_crossover'] == 'bullish':
        return TechnicalSignal(
            signal=SignalType.BUY,
            strength=crossover_info['crossover_strength'],
            reason=f"EMA({fast_period}) bullish crossover above EMA({slow_period}) - Fast: {current_fast:.2f}, Slow: {current_slow:.2f}"
        )
    elif crossover_info['recent_crossover'] == 'bearish':
        return TechnicalSignal(
            signal=SignalType.SELL,
            strength=crossover_info['crossover_strength'],
            reason=f"EMA({fast_period}) bearish crossover below EMA({slow_period}) - Fast: {current_fast:.2f}, Slow: {current_slow:.2f}"
        )
    elif crossover_info['current_position'] == 'above':
        # Fast EMA above slow EMA (uptrend)
        trend_strength = min(abs(current_fast - current_slow) / current_slow * 20, 0.6)
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=trend_strength,
            reason=f"EMA({fast_period}) above EMA({slow_period}) - Uptrend intact - Fast: {current_fast:.2f}, Slow: {current_slow:.2f}"
        )
    elif crossover_info['current_position'] == 'below':
        # Fast EMA below slow EMA (downtrend)
        trend_strength = min(abs(current_fast - current_slow) / current_slow * 20, 0.6)
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=trend_strength,
            reason=f"EMA({fast_period}) below EMA({slow_period}) - Downtrend intact - Fast: {current_fast:.2f}, Slow: {current_slow:.2f}"
        )
    else:
        return TechnicalSignal(
            signal=SignalType.NEUTRAL,
            strength=0.1,
            reason=f"EMA({fast_period}) and EMA({slow_period}) converging - No clear trend"
        )

def interpret_volume_ratio_signal(volume_ratios: List[float], period: int) -> TechnicalSignal:
    """Interpret volume ratio signal for volume confirmation"""
    current_ratio = volume_ratios[-1]
    
    if current_ratio > 2.0:
        # Very high volume
        return TechnicalSignal(
            signal=SignalType.BUY,
            strength=min(current_ratio / 3.0, 1.0),
            reason=f"Very high volume ({current_ratio:.1f}x average) - Strong institutional interest"
        )
    elif current_ratio > 1.5:
        # High volume
        return TechnicalSignal(
            signal=SignalType.BUY,
            strength=min(current_ratio / 2.5, 0.8),
            reason=f"High volume ({current_ratio:.1f}x average) - Increased interest"
        )
    elif current_ratio < 0.5:
        # Very low volume
        return TechnicalSignal(
            signal=SignalType.SELL,
            strength=min((1.0 - current_ratio) * 2, 0.7),
            reason=f"Very low volume ({current_ratio:.1f}x average) - Lack of conviction"
        )
    elif current_ratio < 0.8:
        # Low volume
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=0.3,
            reason=f"Low volume ({current_ratio:.1f}x average) - Weak participation"
        )
    else:
        # Normal volume
        return TechnicalSignal(
            signal=SignalType.NEUTRAL,
            strength=0.2,
            reason=f"Normal volume ({current_ratio:.1f}x average) - Typical activity"
        )

def interpret_volume_trend_signal(volume_trend_data: Dict[str, Any]) -> TechnicalSignal:
    """Interpret volume trend and price-volume relationship"""
    relationship = volume_trend_data['relationship']
    volume_trend = volume_trend_data['volume_trend']
    price_trend = volume_trend_data['price_trend']
    
    if relationship == "bullish_confirmation":
        return TechnicalSignal(
            signal=SignalType.BUY,
            strength=0.8,
            reason=f"Bullish confirmation - Rising prices with increasing volume"
        )
    elif relationship == "bearish_confirmation":
        return TechnicalSignal(
            signal=SignalType.SELL,
            strength=0.8,
            reason=f"Bearish confirmation - Falling prices with increasing volume"
        )
    elif relationship == "weak_bullish":
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=0.3,
            reason=f"Weak bullish - Rising prices but decreasing volume (lack of conviction)"
        )
    elif relationship == "weak_bearish":
        return TechnicalSignal(
            signal=SignalType.HOLD,
            strength=0.3,
            reason=f"Weak bearish - Falling prices but decreasing volume (selling exhaustion?)"
        )
    else:
        return TechnicalSignal(
            signal=SignalType.NEUTRAL,
            strength=0.1,
            reason=f"Neutral volume-price relationship - No clear signal"
        )

# --- Data Fetching Functions ---

def fetch_stock_data(symbol: str, timeframe: str, period_str: str = "1y") -> Optional[Dict[str, List[float]]]:
    """
    Fetch stock data using yfinance
    Returns dict with 'prices' and 'volumes' or None if data cannot be fetched
    """
    try:
        print(f"Fetching stock data for {symbol}, timeframe {timeframe}, period {period_str}")
        
        ticker = yf.Ticker(symbol)
        # Fetch historical data
        data = ticker.history(period=period_str, interval=timeframe)
        
        if data.empty:
            print(f"No data returned for {symbol}")
            return None
            
        prices = data['Close'].tolist()
        volumes = data['Volume'].tolist()
        
        if not prices or not volumes:
            print(f"Empty price or volume list for {symbol}")
            return None
            
        print(f"Successfully fetched {len(prices)} data points for {symbol}")
        return {"prices": prices, "volumes": volumes}
        
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {str(e)}")
        return None

def fetch_crypto_data(symbol: str, timeframe: str, days_history: int = 365) -> Optional[Dict[str, List[float]]]:
    """
    Fetch crypto data using CoinGecko API
    Returns dict with 'prices' and 'volumes' or None if data cannot be fetched
    """
    try:
        print(f"Fetching crypto data for {symbol}, timeframe {timeframe}, days {days_history}")
        
        cg = CoinGeckoAPI()
        
        # Map common symbols to CoinGecko IDs
        symbol_mapping = {
            "BTC-USD": "bitcoin",
            "BTC": "bitcoin", 
            "ETH-USD": "ethereum",
            "ETH": "ethereum",
            "ADA-USD": "cardano",
            "ADA": "cardano",
            "SOL-USD": "solana",
            "SOL": "solana",
            "DOGE-USD": "dogecoin",
            "DOGE": "dogecoin"
        }
        
        coin_id = symbol_mapping.get(symbol.upper())
        if not coin_id:
            print(f"Unknown crypto symbol: {symbol}. Supported symbols: {list(symbol_mapping.keys())}")
            return None
        
        # Fetch market chart data
        market_chart = cg.get_coin_market_chart_by_id(
            id=coin_id, 
            vs_currency='usd', 
            days=days_history
        )
        
        if not market_chart or 'prices' not in market_chart or 'total_volumes' not in market_chart:
            print(f"No price or volume data returned for {symbol}")
            return None
            
        # Extract closing prices and volumes
        prices = [price_point[1] for price_point in market_chart['prices']]
        volumes = [volume_point[1] for volume_point in market_chart['total_volumes']]
        
        if not prices or not volumes:
            print(f"Empty price or volume list for {symbol}")
            return None
            
        print(f"Successfully fetched {len(prices)} data points for {symbol}")
        return {"prices": prices, "volumes": volumes}
        
    except Exception as e:
        print(f"Error fetching crypto data for {symbol}: {str(e)}")
        return None

# --- Core Tool Function ---

def run_technical_analysis_tool(request: TechnicalAnalysisRequest) -> TechnicalAnalysisToolOutput:
    market_data: Optional[Dict[str, List[float]]] = None
    data_fetch_error: Optional[str] = None

    try:
        if request.asset_type == AssetType.STOCK:
            market_data = fetch_stock_data(request.symbol, request.timeframe)
        elif request.asset_type == AssetType.CRYPTO:
            # Map symbol if needed (e.g. BTC-USD to bitcoin for coingecko)
            market_data = fetch_crypto_data(request.symbol, request.timeframe)
        
        if market_data is None or not market_data:
            data_fetch_error = f"Failed to fetch market data for {request.symbol} ({request.asset_type.value}) with timeframe {request.timeframe}."
    except Exception as e:
        data_fetch_error = f"Exception during data fetching for {request.symbol}: {str(e)}"

    indicator_results: List[IndicatorResult] = []
    price_data = market_data["prices"] if market_data else None
    volume_data = market_data["volumes"] if market_data else None
    current_price = price_data[-1] if price_data else None

    if data_fetch_error: # If data fetch failed, still return structure but with data error
        for ind_spec in request.indicators:
            indicator_results.append(IndicatorResult(
                name=ind_spec.name, 
                error=f"Data fetching failed: {data_fetch_error}"
            ))
        return TechnicalAnalysisToolOutput(
            asset_type=request.asset_type,
            symbol=request.symbol,
            timeframe=request.timeframe,
            current_price=None,
            indicators=indicator_results,
            data_fetch_error=data_fetch_error
        )

    for ind_spec in request.indicators:
        values = None
        error_msg: Optional[str] = None
        current_value: Optional[float] = None
        signal: Optional[TechnicalSignal] = None
        
        try:
            params = ind_spec.params if ind_spec.params else {}
            if ind_spec.name.lower() == "sma":
                period = params.get("period", 20) # Default SMA period
                values = sma(price_data, period)
                current_value = values[-1] if values else None
                if values and price_data:
                    signal = interpret_sma_signal(price_data, values, period)
                    
            elif ind_spec.name.lower() == "ema":
                period = params.get("period", 20) # Default EMA period
                values = ema(price_data, period)
                current_value = values[-1] if values else None
                if values and price_data:
                    signal = interpret_ema_signal(price_data, values, period)
                    
            elif ind_spec.name.lower() == "rsi":
                period = params.get("period", 14) # Default RSI period
                values = rsi(price_data, period)
                current_value = values[-1] if values else None
                if values:
                    signal = interpret_rsi_signal(values, period)
            elif ind_spec.name.lower() == "macd":
                fast_period = params.get("fast_period", 12)
                slow_period = params.get("slow_period", 26)
                signal_period = params.get("signal_period", 9)
                values = macd(price_data, fast_period, slow_period, signal_period)
                current_value = values['macd_line'][-1] if values['macd_line'] else None
                if values['macd_line'] and values['signal_line'] and values['histogram']:
                    signal = interpret_macd_signal(values['macd_line'], values['signal_line'], values['histogram'])
                    
            elif ind_spec.name.lower() == "ema_crossover":
                fast_period = params.get("fast_period", 8)
                slow_period = params.get("slow_period", 21)
                ema_fast = ema(price_data, fast_period)
                ema_slow = ema(price_data, slow_period)
                # Store both EMAs in values for reference
                values = {"fast": ema_fast, "slow": ema_slow}
                current_value = ema_fast[-1] if ema_fast else None
                if ema_fast and ema_slow:
                    signal = interpret_ema_crossover_signal(ema_fast, ema_slow, fast_period, slow_period)
            elif ind_spec.name.lower() == "volume_ratio":
                if not volume_data:
                    error_msg = "Volume data not available for volume ratio analysis"
                else:
                    period = params.get("period", 20)
                    values = volume_ratio(volume_data, period)
                    current_value = values[-1] if values else None
                    if values:
                        signal = interpret_volume_ratio_signal(values, period)
            elif ind_spec.name.lower() == "volume_trend":
                if not volume_data or not price_data:
                    error_msg = "Volume and price data not available for volume trend analysis"
                else:
                    values = analyze_volume_trend(volume_data, price_data)
                    current_value = values['avg_volume_change'] if values else None
                    if values:
                        signal = interpret_volume_trend_signal(values)
            else:
                error_msg = f"Unknown indicator: {ind_spec.name}"
                
        except ValueError as ve: # Catch specific errors from indicator functions
             error_msg = f"Calculation error for {ind_spec.name}: {str(ve)}"
        except Exception as e:
            error_msg = f"Unexpected error computing {ind_spec.name}: {str(e)}"
        
        indicator_results.append(IndicatorResult(
            name=ind_spec.name, 
            values=values, 
            error=error_msg,
            current_value=current_value,
            signal=signal
        ))

    return TechnicalAnalysisToolOutput(
        asset_type=request.asset_type,
        symbol=request.symbol,
        timeframe=request.timeframe,
        current_price=current_price,
        indicators=indicator_results,
        data_fetch_error=None # Success
    )

# --- Example Usage ---

if __name__ == "__main__":
    # Professional Trading Setup Example - EMA 8/21 + MACD Confluence
    professional_req = TechnicalAnalysisRequest(
        asset_type=AssetType.STOCK,
        symbol="AAPL",
        timeframe="1d",
        indicators=[
            # EMA crossover system (8/21)
            IndicatorSpec(name="ema_crossover", params={"fast_period": 8, "slow_period": 21}),
            # Standard MACD (12,26,9)
            IndicatorSpec(name="macd", params={"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            # Custom MACD aligned with EMA setup (8,21,9)
            IndicatorSpec(name="macd", params={"fast_period": 8, "slow_period": 21, "signal_period": 9}),
            # RSI for overbought/oversold confirmation
            IndicatorSpec(name="rsi", params={"period": 14}),
        ]
    )
    print("--- Professional Trading Setup Analysis ---")
    professional_result = run_technical_analysis_tool(professional_req)
    print(professional_result.model_dump_json(indent=2))
    print("\n")

    # Example Stock Request
    stock_req = TechnicalAnalysisRequest(
        asset_type=AssetType.STOCK,
        symbol="AAPL",
        timeframe="1d", # Ensure this matches what fetch_stock_data placeholder/implementation supports
        indicators=[
            IndicatorSpec(name="SMA", params={"period": 20}),
            IndicatorSpec(name="EMA", params={"period": 20}),
            IndicatorSpec(name="RSI", params={"period": 14}),
            IndicatorSpec(name="XYZ", params={}), # Unknown indicator
        ]
    )
    print("--- Running Stock Analysis Tool ---")
    stock_result = run_technical_analysis_tool(stock_req)
    print(stock_result.model_dump_json(indent=2))
    print("\n")

    # Example Crypto Request
    crypto_req = TechnicalAnalysisRequest(
        asset_type=AssetType.CRYPTO,
        symbol="BTC-USD", # Ensure this matches what fetch_crypto_data placeholder/implementation supports
        timeframe="1d",
        indicators=[
            IndicatorSpec(name="SMA", params={"period": 50}),
            IndicatorSpec(name="RSI", params={"period": 7}), # Short RSI
            IndicatorSpec(name="EMA", params={"period": 10}),
        ]
    )
    print("--- Running Crypto Analysis Tool ---")
    crypto_result = run_technical_analysis_tool(crypto_req)
    print(crypto_result.model_dump_json(indent=2))
    print("\n")

    # Example Request with data fetching failure (if placeholder returns None for this symbol)
    failed_req = TechnicalAnalysisRequest(
        asset_type=AssetType.STOCK,
        symbol="FAKESYMBOL",
        timeframe="1d",
        indicators=[IndicatorSpec(name="SMA", params={"period": 20})]
    )
    print("--- Running Analysis Tool with potential data failure ---")
    failed_result = run_technical_analysis_tool(failed_req)
    print(failed_result.model_dump_json(indent=2))
    
    # Example Request with insufficient data for calculation (if mock data is too short)
    short_data_req = TechnicalAnalysisRequest(
        asset_type=AssetType.STOCK,
        symbol="AAPL", # Assuming AAPL returns 100 data points from placeholder
        timeframe="1d",
        indicators=[IndicatorSpec(name="SMA", params={"period": 150})] # Period > data length
    )
    print("--- Running Analysis Tool with insufficient data for calculation ---")
    short_data_result = run_technical_analysis_tool(short_data_req)
    print(short_data_result.model_dump_json(indent=2))

