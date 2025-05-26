"""
Sentiment Analysis Manager

Orchestrates multiple sentiment analyzers and provides a unified interface.
Handles configuration, availability checking, and result aggregation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from base import AssetType, SentimentResult, SentimentAggregator
from fear_greed_analyzer import FearGreedAnalyzer
from news_analyzer import NewsSentimentAnalyzer
from economic_analyzer import EconomicSentimentAnalyzer


class SentimentAnalysisRequest(BaseModel):
    """Request for sentiment analysis"""
    asset_type: AssetType
    symbol: str
    sources: List[str] = Field(default_factory=lambda: ["reddit_sentiment", "fear_greed_index"])
    lookback_days: int = Field(default=7, ge=1, le=30)


class SentimentAnalysisResponse(BaseModel):
    """Response from sentiment analysis"""
    asset_type: AssetType
    symbol: str
    lookback_days: int
    overall_sentiment_score: Optional[float] = None
    overall_sentiment_signal: Optional[Any] = None  # SentimentSignal
    sources: List[SentimentResult]
    available_sources: List[str]
    unavailable_sources: List[str]
    error: Optional[str] = None


class SentimentManager:
    """Manages multiple sentiment analyzers"""
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize sentiment manager
        
        Args:
            custom_weights: Custom weights for sentiment aggregation
        """
        # Initialize all analyzers
        self.analyzers = {
            "reddit_sentiment": RedditSentimentAnalyzer(),
            "fear_greed_index": FearGreedAnalyzer(),
            "news_sentiment": NewsSentimentAnalyzer(),
            "economic_sentiment": EconomicSentimentAnalyzer()
        }
        
        # Initialize aggregator
        self.aggregator = SentimentAggregator(custom_weights)
    
    def analyze(self, request: SentimentAnalysisRequest) -> SentimentAnalysisResponse:
        """
        Run sentiment analysis for the requested sources
        
        Args:
            request: Sentiment analysis request
            
        Returns:
            Sentiment analysis response with results from all sources
        """
        try:
            # Check which sources are available
            available_sources = []
            unavailable_sources = []
            
            for source in request.sources:
                if source in self.analyzers:
                    if self.analyzers[source].is_available():
                        available_sources.append(source)
                    else:
                        unavailable_sources.append(source)
                else:
                    unavailable_sources.append(source)
            
            # Run analysis on available sources
            results = []
            for source in available_sources:
                try:
                    analyzer = self.analyzers[source]
                    result = analyzer.analyze(
                        symbol=request.symbol,
                        asset_type=request.asset_type,
                        lookback_days=request.lookback_days
                    )
                    results.append(result)
                    
                except Exception as e:
                    # Create error result for this source
                    error_result = SentimentResult(
                        source=source,
                        error=f"Error in {source}: {str(e)}"
                    )
                    results.append(error_result)
            
            # Aggregate results
            overall_score, overall_signal = self.aggregator.aggregate(results)
            
            return SentimentAnalysisResponse(
                asset_type=request.asset_type,
                symbol=request.symbol,
                lookback_days=request.lookback_days,
                overall_sentiment_score=overall_score,
                overall_sentiment_signal=overall_signal,
                sources=results,
                available_sources=available_sources,
                unavailable_sources=unavailable_sources
            )
            
        except Exception as e:
            return SentimentAnalysisResponse(
                asset_type=request.asset_type,
                symbol=request.symbol,
                lookback_days=request.lookback_days,
                sources=[],
                available_sources=[],
                unavailable_sources=request.sources,
                error=f"Critical error in sentiment analysis: {str(e)}"
            )
    
    def get_available_sources(self) -> Dict[str, bool]:
        """Get availability status of all sources"""
        return {
            name: analyzer.is_available() 
            for name, analyzer in self.analyzers.items()
        }
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get configuration information for all analyzers"""
        config_info = {}
        
        for name, analyzer in self.analyzers.items():
            try:
                if hasattr(analyzer, 'get_configuration_info'):
                    config_info[name] = analyzer.get_configuration_info()
                else:
                    config_info[name] = {
                        "available": analyzer.is_available(),
                        "required_config": analyzer.get_required_config()
                    }
            except Exception as e:
                config_info[name] = {"error": str(e)}
        
        return config_info
    
    def get_setup_instructions(self) -> Dict[str, str]:
        """Get setup instructions for unavailable sources"""
        instructions = {}
        
        for name, analyzer in self.analyzers.items():
            if not analyzer.is_available():
                required_config = analyzer.get_required_config()
                if required_config:
                    if name == "news_sentiment":
                        instructions[name] = (
                            "1. Visit https://newsapi.org/\n"
                            "2. Sign up for free account (1,000 requests/day)\n"
                            "3. Get your API key\n"
                            "4. Set environment variable: export NEWSAPI_KEY='your_key'"
                        )
                    elif name == "economic_sentiment":
                        instructions[name] = (
                            "1. Visit https://fred.stlouisfed.org/docs/api/api_key.html\n"
                            "2. Request free API key (unlimited requests)\n"
                            "3. Set environment variable: export FRED_API_KEY='your_key'"
                        )
                    else:
                        instructions[name] = f"Required config: {', '.join(required_config)}"
        
        return instructions
    
    def test_all_sources(self, test_symbol: str = "AAPL", test_asset_type: AssetType = AssetType.STOCK) -> Dict[str, Any]:
        """Test all sentiment sources with a sample request"""
        test_results = {}
        
        for name, analyzer in self.analyzers.items():
            try:
                if analyzer.is_available():
                    result = analyzer.analyze(
                        symbol=test_symbol,
                        asset_type=test_asset_type,
                        lookback_days=3  # Short test
                    )
                    test_results[name] = {
                        "status": "success" if result.error is None else "error",
                        "sentiment_score": result.sentiment_score,
                        "data_points": result.data_points,
                        "error": result.error
                    }
                else:
                    test_results[name] = {
                        "status": "unavailable",
                        "reason": f"Required config: {analyzer.get_required_config()}"
                    }
                    
            except Exception as e:
                test_results[name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return test_results


# Convenience function for simple usage
def analyze_sentiment(symbol: str, 
                     asset_type: AssetType, 
                     sources: Optional[List[str]] = None,
                     lookback_days: int = 7) -> SentimentAnalysisResponse:
    """
    Convenience function for sentiment analysis
    
    Args:
        symbol: Asset symbol to analyze
        asset_type: Type of asset (STOCK or CRYPTO)
        sources: List of sentiment sources to use (default: free sources only)
        lookback_days: Number of days to look back
        
    Returns:
        Sentiment analysis response
    """
    if sources is None:
        # Default to free sources that don't require API keys
        sources = ["reddit_sentiment", "fear_greed_index"]
    
    manager = SentimentManager()
    request = SentimentAnalysisRequest(
        asset_type=asset_type,
        symbol=symbol,
        sources=sources,
        lookback_days=lookback_days
    )
    
    return manager.analyze(request) 