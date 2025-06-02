"""
Base classes and interfaces for sentiment analysis modules
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class AssetType(str, Enum):
    STOCK = "STOCK"
    CRYPTO = "CRYPTO"
    ETF = "ETF"
    INDEX = "INDEX"

class SentimentType(str, Enum):
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"


class SentimentSignal(BaseModel):
    sentiment: SentimentType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level from 0.0 to 1.0")
    reason: str = Field(..., description="Human-readable explanation of the sentiment")


class SentimentResult(BaseModel):
    source: str
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Sentiment score from -1.0 (very bearish) to 1.0 (very bullish)")
    sentiment_signal: Optional[SentimentSignal] = None
    data_points: Optional[int] = Field(None, ge=0, description="Number of data points analyzed")
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SentimentAnalyzer(ABC):
    """Abstract base class for all sentiment analyzers"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze(self, symbol: str, asset_type: AssetType, **kwargs) -> SentimentResult:
        """
        Analyze sentiment for a given symbol
        
        Args:
            symbol: The asset symbol (e.g., 'AAPL', 'BTC')
            asset_type: Type of asset (STOCK or CRYPTO)
            **kwargs: Additional parameters specific to the analyzer
            
        Returns:
            SentimentResult with analysis results
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the analyzer is available (e.g., API keys are set)"""
        pass
    
    def get_required_config(self) -> List[str]:
        """Return list of required configuration keys"""
        return []


def interpret_sentiment_score(score: float, source_type: str) -> SentimentSignal:
    """Convert numerical sentiment score to actionable sentiment signal"""
    
    if score >= 0.6:
        return SentimentSignal(
            sentiment=SentimentType.VERY_BULLISH,
            confidence=min(score, 1.0),
            reason=f"Very bullish {source_type} sentiment (score: {score:.2f}) - Strong positive indicators"
        )
    elif score >= 0.2:
        return SentimentSignal(
            sentiment=SentimentType.BULLISH,
            confidence=score * 0.8,
            reason=f"Bullish {source_type} sentiment (score: {score:.2f}) - Positive indicators outweigh negative"
        )
    elif score >= -0.2:
        return SentimentSignal(
            sentiment=SentimentType.NEUTRAL,
            confidence=0.3,
            reason=f"Neutral {source_type} sentiment (score: {score:.2f}) - Mixed or balanced indicators"
        )
    elif score >= -0.6:
        return SentimentSignal(
            sentiment=SentimentType.BEARISH,
            confidence=abs(score) * 0.8,
            reason=f"Bearish {source_type} sentiment (score: {score:.2f}) - Negative indicators outweigh positive"
        )
    else:
        return SentimentSignal(
            sentiment=SentimentType.VERY_BEARISH,
            confidence=min(abs(score), 1.0),
            reason=f"Very bearish {source_type} sentiment (score: {score:.2f}) - Strong negative indicators"
        )


class SentimentAggregator:
    """Aggregates sentiment from multiple sources"""
    
    def __init__(self, source_weights: Optional[Dict[str, float]] = None):
        self.source_weights = source_weights or {
            "news_sentiment": 0.30,
            "fear_greed_index": 0.25,
            "economic_sentiment": 0.20
        }
    
    def aggregate(self, sentiment_results: List[SentimentResult]) -> tuple[float, SentimentSignal]:
        """Aggregate multiple sentiment scores into overall sentiment"""
        
        valid_results = [r for r in sentiment_results if r.sentiment_score is not None and r.error is None]
        
        if not valid_results:
            return 0.0, SentimentSignal(
                sentiment=SentimentType.NEUTRAL,
                confidence=0.0,
                reason="No valid sentiment data available for aggregation"
            )
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in valid_results:
            weight = self.source_weights.get(result.source, 0.15)  # Default weight
            weighted_sum += result.sentiment_score * weight
            total_weight += weight
        
        if total_weight == 0:
            overall_score = 0.0
        else:
            overall_score = round(weighted_sum / total_weight, 3)
        
        # Calculate confidence based on agreement between sources
        scores = [r.sentiment_score for r in valid_results]
        score_variance = sum((s - overall_score) ** 2 for s in scores) / len(scores)
        agreement_confidence = max(0.0, 1.0 - score_variance * 2)  # Lower variance = higher confidence
        
        overall_signal = interpret_sentiment_score(overall_score, "aggregated")
        overall_signal.confidence = round(min(overall_signal.confidence * agreement_confidence, 1.0), 3)
        
        return overall_score, overall_signal 