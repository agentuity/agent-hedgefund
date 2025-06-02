"""
Technical Analyst

Clean interface for technical analysis using the technical tools ecosystem.
Provides comprehensive technical indicator analysis with actionable signals.
"""

from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

from agents.hedge_fund.tools.technical import (
    TechnicalAnalysisManager, analyze_technical_indicators,
    TechnicalAnalysisResult
)
from agents.hedge_fund.tools.sentiment.base import AssetType

logger = logging.getLogger(__name__)

# --- Data Models for Trade Decision Agent Compatibility ---

class TechnicalSignal(str, Enum):
    """Technical signal types"""
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"
    UNCLEAR = "UNCLEAR"

@dataclass
class IndicatorSpec:
    """Specification for a technical indicator"""
    name: str
    params: Dict[str, Any]

@dataclass
class SignalOutput:
    """Output from a technical signal"""
    signal: TechnicalSignal
    strength: float  # 0.0-1.0
    reason: str

@dataclass 
class IndicatorOutput:
    """Output from a technical indicator"""
    name: str
    values: List[float]
    signal: Optional[SignalOutput] = None

class TechnicalAnalysisRequest(BaseModel):
    """Request for technical analysis"""
    asset_type: AssetType
    symbol: str
    timeframe: str = "1d"
    indicators: List[IndicatorSpec] = Field(default_factory=list)

class TechnicalAnalysisToolOutput(BaseModel):
    """Output from technical analysis tool"""
    symbol: str
    asset_type: AssetType
    indicators: List[IndicatorOutput]
    summary: str
    overall_signal: TechnicalSignal
    confidence: float
    current_price: Optional[float] = None
    timestamp: str

def run_technical_analysis_tool(request: TechnicalAnalysisRequest) -> TechnicalAnalysisToolOutput:
    """
    Run technical analysis tool compatible with trade decision agent
    Maps the new technical analysis to the old interface format
    """
    try:
        # Use the new technical analysis system
        analysis = analyze_technical_indicators(request.symbol, request.asset_type)
        
        if not analysis:
            return TechnicalAnalysisToolOutput(
                symbol=request.symbol,
                asset_type=request.asset_type,
                indicators=[],
                summary=f"Unable to analyze {request.symbol}",
                overall_signal=TechnicalSignal.UNCLEAR,
                confidence=0.0,
                current_price=None,
                timestamp=str(analysis.timestamp) if analysis else ""
            )
        
        # Convert to old format
        indicators = []
        
        # Map indicators to the old format - FIX: properly access signal direction
        if analysis.indicators.sma_20:
            sma_direction = analysis.signals.sma_signal.get('direction')
            signal_strength = analysis.signals.sma_signal.get('confidence', 0.5)
            if sma_direction and sma_direction.value == "bullish":
                signal_type = TechnicalSignal.BUY
            elif sma_direction and sma_direction.value == "bearish":
                signal_type = TechnicalSignal.SELL
            else:
                signal_type = TechnicalSignal.HOLD
            
            indicators.append(IndicatorOutput(
                name="sma",
                values=analysis.indicators.sma_20,
                signal=SignalOutput(signal_type, signal_strength, f"SMA trend: {sma_direction.value if sma_direction else 'neutral'}")
            ))
        
        if analysis.indicators.ema_8:
            ema_direction = analysis.signals.ema_signal.get('direction')
            signal_strength = analysis.signals.ema_signal.get('confidence', 0.5)
            if ema_direction and ema_direction.value == "bullish":
                signal_type = TechnicalSignal.BUY
            elif ema_direction and ema_direction.value == "bearish":
                signal_type = TechnicalSignal.SELL
            else:
                signal_type = TechnicalSignal.HOLD
                
            indicators.append(IndicatorOutput(
                name="ema_crossover",
                values=analysis.indicators.ema_8,
                signal=SignalOutput(signal_type, signal_strength, f"EMA crossover: {ema_direction.value if ema_direction else 'neutral'}")
            ))
        
        if analysis.indicators.rsi_14:
            rsi_direction = analysis.signals.rsi_signal.get('direction')
            signal_strength = analysis.signals.rsi_signal.get('confidence', 0.5)
            if rsi_direction and rsi_direction.value == "bullish":
                signal_type = TechnicalSignal.BUY
            elif rsi_direction and rsi_direction.value == "bearish":
                signal_type = TechnicalSignal.SELL
            else:
                signal_type = TechnicalSignal.HOLD
                
            indicators.append(IndicatorOutput(
                name="rsi",
                values=analysis.indicators.rsi_14,
                signal=SignalOutput(signal_type, signal_strength, f"RSI: {rsi_direction.value if rsi_direction else 'neutral'}")
            ))
        
        # Add MACD indicator if available
        if analysis.indicators.macd and 'macd_line' in analysis.indicators.macd:
            macd_direction = analysis.signals.macd_signal.get('direction')
            signal_strength = analysis.signals.macd_signal.get('confidence', 0.5)
            if macd_direction and macd_direction.value == "bullish":
                signal_type = TechnicalSignal.BUY
            elif macd_direction and macd_direction.value == "bearish":
                signal_type = TechnicalSignal.SELL
            else:
                signal_type = TechnicalSignal.HOLD
                
            indicators.append(IndicatorOutput(
                name="macd",
                values=analysis.indicators.macd['macd_line'],
                signal=SignalOutput(signal_type, signal_strength, f"MACD: {macd_direction.value if macd_direction else 'neutral'}")
            ))
        
        # Determine overall signal - FIX: properly access direction
        if analysis.overall_direction.value == "bullish":
            overall_signal = TechnicalSignal.BUY
        elif analysis.overall_direction.value == "bearish":
            overall_signal = TechnicalSignal.SELL
        else:
            overall_signal = TechnicalSignal.HOLD
        
        return TechnicalAnalysisToolOutput(
            symbol=request.symbol,
            asset_type=request.asset_type,
            indicators=indicators,
            summary=analysis.summary,
            overall_signal=overall_signal,
            confidence=analysis.overall_confidence,
            current_price=analysis.market_data.current_price,
            timestamp=str(analysis.timestamp)
        )
        
    except Exception as e:
        logger.error(f"Error in technical analysis tool: {e}")
        return TechnicalAnalysisToolOutput(
            symbol=request.symbol,
            asset_type=request.asset_type,
            indicators=[],
            summary=f"Analysis failed: {str(e)}",
            overall_signal=TechnicalSignal.UNCLEAR,
            confidence=0.0,
            current_price=None,
            timestamp=""
        )

# --- Enhanced Technical Analysis Interface ---

class TechnicalAnalysis:
    """
    Clean interface for technical analysis
    Provides professional-grade technical indicator analysis
    """
    
    def __init__(self):
        self.manager = TechnicalAnalysisManager()
        self.logger = logging.getLogger(__name__)
    
    def analyze_asset(self, symbol: str, asset_type: AssetType) -> Optional[TechnicalAnalysisResult]:
        """
        Perform comprehensive technical analysis on an asset
        Returns complete analysis with indicators, signals, and summary
        """
        self.logger.info(f"🔍 Analyzing technical indicators for {symbol}")
        return self.manager.analyze_asset(symbol, asset_type)
    
    def get_trading_signals(self, symbol: str, asset_type: AssetType) -> Dict[str, Any]:
        """
        Get focused trading signals for decision making
        Returns simplified signals for trading decisions
        """
        analysis = self.analyze_asset(symbol, asset_type)
        if not analysis:
            return {
                'status': 'error',
                'message': f'Unable to analyze {symbol}',
                'signals': {}
            }
        
        return {
            'status': 'success',
            'symbol': symbol,
            'overall_direction': analysis.overall_direction.value,
            'overall_confidence': analysis.overall_confidence,
            'summary': analysis.summary,
            'signals': {
                'sma': analysis.signals.sma_signal,
                'ema': analysis.signals.ema_signal,
                'rsi': analysis.signals.rsi_signal,
                'macd': analysis.signals.macd_signal,
                'crossover': analysis.signals.crossover_signal
            },
            'current_price': analysis.market_data.current_price,
            'data_points': len(analysis.market_data.prices)
        }
    
    def get_key_levels(self, symbol: str, asset_type: AssetType) -> Dict[str, Any]:
        """
        Get key technical levels for the asset
        Returns support/resistance and moving average levels
        """
        analysis = self.analyze_asset(symbol, asset_type)
        if not analysis:
            return {'status': 'error', 'message': f'Unable to analyze {symbol}'}
        
        # Get latest values from indicators
        current_price = analysis.market_data.current_price
        sma_20 = analysis.indicators.sma_20[-1] if analysis.indicators.sma_20 else None
        ema_8 = analysis.indicators.ema_8[-1] if analysis.indicators.ema_8 else None
        ema_21 = analysis.indicators.ema_21[-1] if analysis.indicators.ema_21 else None
        rsi = analysis.indicators.rsi_14[-1] if analysis.indicators.rsi_14 else None
        
        return {
            'status': 'success',
            'symbol': symbol,
            'current_price': current_price,
            'key_levels': {
                'sma_20': sma_20,
                'ema_8': ema_8,
                'ema_21': ema_21,
                'rsi_level': rsi
            },
            'price_vs_levels': {
                'above_sma_20': current_price > sma_20 if sma_20 else None,
                'above_ema_8': current_price > ema_8 if ema_8 else None,
                'above_ema_21': current_price > ema_21 if ema_21 else None,
                'ema_8_above_21': ema_8 > ema_21 if ema_8 and ema_21 else None
            }
        }

def run_technical_analysis(symbol: str, asset_type: str = "stock") -> Dict[str, Any]:
    """
    Legacy support function for existing code
    Maps old string asset_type to new AssetType enum
    """
    # Convert string to AssetType enum
    if asset_type.lower() == "crypto":
        asset_enum = AssetType.CRYPTO
    else:
        asset_enum = AssetType.STOCK
    
    analyzer = TechnicalAnalysis()
    return analyzer.get_trading_signals(symbol, asset_enum)

def get_technical_summary(symbol: str, asset_type: str = "stock") -> str:
    """
    Get a brief technical summary for quick reference
    """
    analysis_result = run_technical_analysis(symbol, asset_type)
    
    if analysis_result['status'] == 'error':
        return f"Technical analysis failed for {symbol}: {analysis_result['message']}"
    
    return analysis_result.get('summary', f"No summary available for {symbol}")

# --- Convenience Functions ---

def analyze_stock(symbol: str) -> Optional[TechnicalAnalysisResult]:
    """Quick stock analysis"""
    return analyze_technical_indicators(symbol, AssetType.STOCK)

def analyze_crypto(symbol: str) -> Optional[TechnicalAnalysisResult]:
    """Quick crypto analysis"""
    return analyze_technical_indicators(symbol, AssetType.CRYPTO)

def get_ema_crossover_status(symbol: str, asset_type: AssetType) -> Dict[str, Any]:
    """
    Get focused EMA crossover analysis (popular trading signal)
    """
    analyzer = TechnicalAnalysis()
    analysis = analyzer.analyze_asset(symbol, asset_type)
    
    if not analysis:
        return {'status': 'error', 'message': f'Unable to analyze {symbol}'}
    
    crossover_signal = analysis.signals.crossover_signal
    ema_crossover = analysis.indicators.ema_crossover
    
    return {
        'status': 'success',
        'symbol': symbol,
        'crossover_signal': crossover_signal,
        'crossover_details': ema_crossover,
        'ema_8': analysis.indicators.ema_8[-1] if analysis.indicators.ema_8 else None,
        'ema_21': analysis.indicators.ema_21[-1] if analysis.indicators.ema_21 else None,
        'current_price': analysis.market_data.current_price
    }

# --- Public Interface ---

__all__ = [
    "TechnicalAnalysis",
    "run_technical_analysis", 
    "get_technical_summary",
    "analyze_stock",
    "analyze_crypto", 
    "get_ema_crossover_status",
    "TechnicalAnalysisRequest",
    "TechnicalAnalysisToolOutput", 
    "run_technical_analysis_tool",
    "AssetType",
    "IndicatorSpec",
    "TechnicalSignal",
    "SignalOutput",
    "IndicatorOutput"
]

