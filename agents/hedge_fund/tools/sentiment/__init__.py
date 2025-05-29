"""
Sentiment Analysis Package

This package provides modular sentiment analysis capabilities for financial markets.
Each sentiment source is implemented as a separate module following the SentimentAnalyzer interface.
"""

from .base import (
    AssetType,
    SentimentType,
    SentimentSignal,
    SentimentResult,
    SentimentAnalyzer,
    SentimentAggregator,
    interpret_sentiment_score
)

__all__ = [
    'AssetType',
    'SentimentType', 
    'SentimentSignal',
    'SentimentResult',
    'SentimentAnalyzer',
    'SentimentAggregator',
    'interpret_sentiment_score'
] 