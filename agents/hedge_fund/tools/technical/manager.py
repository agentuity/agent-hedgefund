"""
Technical Analysis Manager

Orchestrates the complete technical analysis workflow:
1. Fetch market data
2. Calculate technical indicators  
3. Interpret signals
4. Provide comprehensive analysis
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .data_fetcher import fetch_market_data, MarketData, validate_market_data
from .indicators import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    detect_ema_crossover, calculate_volume_indicators
)
from .signal_interpreter import (
    interpret_sma_signal, interpret_ema_signal, interpret_rsi_signal,
    interpret_macd_signal, interpret_ema_crossover_signal,
    SignalDirection, SignalStrength
)
from agents.hedge_fund.tools.sentiment.base import AssetType

# Set up logger
logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicatorResults:
    """Container for all technical indicator calculations"""
    sma_20: List[float]
    ema_8: List[float]
    ema_21: List[float]
    rsi_14: List[float]
    macd: Dict[str, List[float]]
    volume_analysis: Dict[str, Any]
    ema_crossover: Dict[str, Any]

@dataclass
class TechnicalSignals:
    """Container for all interpreted signals"""
    sma_signal: Dict[str, Any]
    ema_signal: Dict[str, Any]
    rsi_signal: Dict[str, Any]
    macd_signal: Dict[str, Any]
    crossover_signal: Dict[str, Any]
    
    def get_overall_direction(self) -> SignalDirection:
        """Calculate overall signal direction from all indicators"""
        bullish_count = 0
        bearish_count = 0
        total_confidence = 0
        signal_count = 0
        
        for signal_name in ['sma_signal', 'ema_signal', 'rsi_signal', 'macd_signal', 'crossover_signal']:
            signal = getattr(self, signal_name)
            if signal and 'direction' in signal and 'confidence' in signal:
                direction = signal['direction']
                confidence = signal['confidence']
                
                if direction == SignalDirection.BULLISH:
                    bullish_count += confidence
                elif direction == SignalDirection.BEARISH:
                    bearish_count += confidence
                    
                total_confidence += confidence
                signal_count += 1
        
        if signal_count == 0:
            return SignalDirection.NEUTRAL
            
        if bullish_count > bearish_count:
            return SignalDirection.BULLISH
        elif bearish_count > bullish_count:
            return SignalDirection.BEARISH
        else:
            return SignalDirection.NEUTRAL
    
    def get_overall_confidence(self) -> float:
        """Calculate overall confidence score"""
        confidences = []
        for signal_name in ['sma_signal', 'ema_signal', 'rsi_signal', 'macd_signal', 'crossover_signal']:
            signal = getattr(self, signal_name)
            if signal and 'confidence' in signal:
                confidences.append(signal['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0

@dataclass
class TechnicalAnalysisResult:
    """Complete technical analysis result"""
    symbol: str
    asset_type: AssetType
    market_data: MarketData
    indicators: TechnicalIndicatorResults
    signals: TechnicalSignals
    overall_direction: SignalDirection
    overall_confidence: float
    summary: str
    timestamp: str

class TechnicalAnalysisManager:
    """
    Manager class for coordinating technical analysis workflow
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_asset(self, symbol: str, asset_type: AssetType) -> Optional[TechnicalAnalysisResult]:
        """
        Perform complete technical analysis on an asset
        """
        try:
            self.logger.info(f"🔍 Starting technical analysis for {symbol} ({asset_type.value})")
            
            # Step 1: Fetch market data
            market_data = fetch_market_data(symbol, asset_type)
            if not market_data:
                self.logger.error(f"❌ Failed to fetch market data for {symbol}")
                return None
            
            # Step 2: Validate data sufficiency
            if not validate_market_data(market_data, min_periods=50):
                self.logger.error(f"❌ Insufficient market data for {symbol}")
                return None
            
            # Step 3: Calculate technical indicators
            indicators = self._calculate_indicators(market_data)
            if not indicators:
                self.logger.error(f"❌ Failed to calculate indicators for {symbol}")
                return None
            
            # Step 4: Interpret signals
            signals = self._interpret_signals(market_data, indicators)
            
            # Step 5: Generate overall assessment
            overall_direction = signals.get_overall_direction()
            overall_confidence = signals.get_overall_confidence()
            
            # Step 6: Create summary
            summary = self._generate_summary(symbol, signals, overall_direction, overall_confidence)
            
            self.logger.info(f"✅ Technical analysis completed for {symbol}")
            
            return TechnicalAnalysisResult(
                symbol=symbol,
                asset_type=asset_type,
                market_data=market_data,
                indicators=indicators,
                signals=signals,
                overall_direction=overall_direction,
                overall_confidence=overall_confidence,
                summary=summary,
                timestamp="current"  # Would use actual timestamp in production
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error in technical analysis for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, market_data: MarketData) -> Optional[TechnicalIndicatorResults]:
        """Calculate all technical indicators"""
        try:
            prices = market_data.prices
            volumes = market_data.volumes
            
            # Calculate moving averages
            sma_20 = calculate_sma(prices, 20)
            ema_8 = calculate_ema(prices, 8)
            ema_21 = calculate_ema(prices, 21)
            
            # Calculate momentum indicators
            rsi_14 = calculate_rsi(prices, 14)
            macd = calculate_macd(prices)
            
            # Calculate volume indicators
            volume_analysis = calculate_volume_indicators(volumes, prices)
            
            # Calculate EMA crossover
            ema_crossover = detect_ema_crossover(ema_8, ema_21)
            
            return TechnicalIndicatorResults(
                sma_20=sma_20,
                ema_8=ema_8,
                ema_21=ema_21,
                rsi_14=rsi_14,
                macd=macd,
                volume_analysis=volume_analysis,
                ema_crossover=ema_crossover
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating indicators: {e}")
            return None
    
    def _interpret_signals(self, market_data: MarketData, indicators: TechnicalIndicatorResults) -> TechnicalSignals:
        """Interpret all technical indicators into actionable signals"""
        try:
            prices = market_data.prices
            
            # Interpret each indicator
            sma_signal = interpret_sma_signal(prices, indicators.sma_20, 20)
            ema_signal = interpret_ema_signal(prices, indicators.ema_8, indicators.ema_21, 8, 21)
            rsi_signal = interpret_rsi_signal(indicators.rsi_14)
            macd_signal = interpret_macd_signal(indicators.macd)
            crossover_signal = interpret_ema_crossover_signal(indicators.ema_crossover)
            
            return TechnicalSignals(
                sma_signal=sma_signal,
                ema_signal=ema_signal,
                rsi_signal=rsi_signal,
                macd_signal=macd_signal,
                crossover_signal=crossover_signal
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error interpreting signals: {e}")
            # Return neutral signals on error
            neutral_signal = {
                'direction': SignalDirection.NEUTRAL,
                'strength': SignalStrength.WEAK,
                'confidence': 0.0,
                'explanation': f"Error interpreting signal: {e}"
            }
            return TechnicalSignals(
                sma_signal=neutral_signal,
                ema_signal=neutral_signal,
                rsi_signal=neutral_signal,
                macd_signal=neutral_signal,
                crossover_signal=neutral_signal
            )
    
    def _generate_summary(self, symbol: str, signals: TechnicalSignals, 
                         overall_direction: SignalDirection, overall_confidence: float) -> str:
        """Generate human-readable summary of the analysis"""
        
        # Count signal strengths
        strong_signals = []
        moderate_signals = []
        
        for signal_name, signal in [
            ("SMA", signals.sma_signal),
            ("EMA", signals.ema_signal), 
            ("RSI", signals.rsi_signal),
            ("MACD", signals.macd_signal),
            ("Crossover", signals.crossover_signal)
        ]:
            if signal and 'strength' in signal and 'direction' in signal:
                strength = signal['strength']
                direction = signal['direction']
                
                if strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
                    strong_signals.append(f"{signal_name} ({direction.value})")
                elif strength == SignalStrength.MODERATE:
                    moderate_signals.append(f"{signal_name} ({direction.value})")
        
        # Build summary
        summary_parts = [f"Technical Analysis for {symbol}:"]
        
        if overall_direction == SignalDirection.BULLISH:
            summary_parts.append(f"🟢 BULLISH outlook with {overall_confidence:.1%} confidence")
        elif overall_direction == SignalDirection.BEARISH:
            summary_parts.append(f"🔴 BEARISH outlook with {overall_confidence:.1%} confidence")
        else:
            summary_parts.append(f"🟡 NEUTRAL outlook with {overall_confidence:.1%} confidence")
        
        if strong_signals:
            summary_parts.append(f"Strong signals: {', '.join(strong_signals)}")
        
        if moderate_signals:
            summary_parts.append(f"Moderate signals: {', '.join(moderate_signals)}")
        
        return " | ".join(summary_parts)

# --- Convenience Functions ---

def analyze_technical_indicators(symbol: str, asset_type: AssetType) -> Optional[TechnicalAnalysisResult]:
    """
    Convenience function for quick technical analysis
    """
    manager = TechnicalAnalysisManager()
    return manager.analyze_asset(symbol, asset_type) 