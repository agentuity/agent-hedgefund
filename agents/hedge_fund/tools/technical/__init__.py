"""
Technical Analysis Tools Package

Contains specialized technical analysis components:
- indicators: Raw technical indicator calculations
- data_fetcher: Market data retrieval
- signal_interpreter: Convert calculations to actionable signals
- manager: Orchestrate technical analysis workflow
"""

from agents.hedge_fund.tools.technical.manager import TechnicalAnalysisManager, analyze_technical_indicators, TechnicalAnalysisResult
from agents.hedge_fund.tools.technical.indicators import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    detect_ema_crossover, calculate_volume_indicators
)
from agents.hedge_fund.tools.technical.signal_interpreter import (
    interpret_sma_signal, interpret_ema_signal, interpret_rsi_signal,
    interpret_macd_signal, interpret_ema_crossover_signal
)
from agents.hedge_fund.tools.technical.data_fetcher import fetch_market_data, MarketData

__all__ = [
    "TechnicalAnalysisManager",
    "analyze_technical_indicators",
    "TechnicalAnalysisResult",
    "calculate_sma",
    "calculate_ema", 
    "calculate_rsi",
    "calculate_macd",
    "detect_ema_crossover",
    "calculate_volume_indicators",
    "interpret_sma_signal",
    "interpret_ema_signal",
    "interpret_rsi_signal", 
    "interpret_macd_signal",
    "interpret_ema_crossover_signal",
    "fetch_market_data",
    "MarketData"
] 