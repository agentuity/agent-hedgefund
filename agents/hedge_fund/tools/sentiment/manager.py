"""
Sentiment Analysis Manager

Orchestrates multiple sentiment analyzers and provides a unified interface.
Handles configuration, availability checking, and result aggregation.
Enhanced with LLM-powered news analysis support.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from agents.hedge_fund.tools.sentiment.base import AssetType, SentimentResult, SentimentAggregator
from agents.hedge_fund.tools.sentiment.fear_greed_analyzer import FearGreedAnalyzer
from agents.hedge_fund.tools.sentiment.news_analyzer import NewsSentimentAnalyzer
from agents.hedge_fund.tools.sentiment.economic_analyzer import EconomicSentimentAnalyzer


class SentimentAnalysisRequest(BaseModel):
    """Request for sentiment analysis"""
    asset_type: AssetType
    symbol: str
    sources: List[str] = Field(default_factory=lambda: ["fear_greed_index"])
    lookback_days: int = Field(default=7, ge=1, le=30)
    use_llm: bool = Field(default=True, description="Enable LLM-powered analysis for news sentiment")


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
    enhanced_analysis: Optional[Dict[str, Any]] = None  # LLM insights when available
    error: Optional[str] = None


class SentimentManager:
    """Manages multiple sentiment analyzers with enhanced LLM capabilities"""
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None, enable_llm: bool = True):
        """
        Initialize sentiment manager
        
        Args:
            custom_weights: Custom weights for sentiment aggregation
            enable_llm: Enable LLM-powered analysis globally
        """
        # Initialize all analyzers with enhanced capabilities
        self.analyzers = {
            "fear_greed_index": FearGreedAnalyzer(),
            "news_sentiment": NewsSentimentAnalyzer(
                max_articles=15,
                use_llm=enable_llm
            ),
            "economic_sentiment": EconomicSentimentAnalyzer()
        }
        
        # Initialize aggregator with enhanced weights for LLM analysis
        enhanced_weights = custom_weights or {
            "news_sentiment": 0.40,      # Higher weight for enhanced news analysis
            "fear_greed_index": 0.35,
            "economic_sentiment": 0.25
        }
        
        self.aggregator = SentimentAggregator(enhanced_weights)
        self.enable_llm = enable_llm
    
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
            enhanced_insights = None
            
            for source in available_sources:
                try:
                    analyzer = self.analyzers[source]
                    
                    # Pass LLM settings for news analyzer
                    kwargs = {
                        'lookback_days': request.lookback_days
                    }
                    if source == "news_sentiment" and hasattr(request, 'use_llm'):
                        kwargs['use_llm_override'] = request.use_llm
                    
                    result = analyzer.analyze(
                        symbol=request.symbol,
                        asset_type=request.asset_type,
                        **kwargs
                    )
                    results.append(result)
                    
                    # Extract enhanced insights from news analysis if available
                    if (source == "news_sentiment" and 
                        result.metadata and 
                        result.metadata.get("analysis_method") == "enhanced_llm"):
                        enhanced_insights = {
                            "key_themes": result.metadata.get("key_themes", []),
                            "market_moving_events": result.metadata.get("market_moving_events", []),
                            "risk_factors": result.metadata.get("risk_factors", []),
                            "opportunities": result.metadata.get("opportunities", []),
                            "llm_confidence": result.metadata.get("llm_confidence", 0.0)
                        }
                    
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
                unavailable_sources=unavailable_sources,
                enhanced_analysis=enhanced_insights
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
        
        # Add global LLM status
        config_info["global_settings"] = {
            "llm_enabled": self.enable_llm,
            "enhanced_news_analysis": config_info.get("news_sentiment", {}).get("llm_available", False)
        }
        
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
                            "4. Set environment variable: export NEWSAPI_KEY='your_key'\n"
                            "5. (Optional) Set OPENAI_API_KEY for enhanced LLM analysis"
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
                    # Test with enhanced features if applicable
                    kwargs = {'lookback_days': 3}
                    if name == "news_sentiment":
                        kwargs['use_llm_override'] = True
                    
                    result = analyzer.analyze(
                        symbol=test_symbol,
                        asset_type=test_asset_type,
                        **kwargs
                    )
                    test_results[name] = {
                        "status": "success" if result.error is None else "error",
                        "sentiment_score": result.sentiment_score,
                        "data_points": result.data_points,
                        "error": result.error,
                        "analysis_method": result.metadata.get("analysis_method") if result.metadata else "basic"
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

    def analyze_with_insights(self, symbol: str, asset_type: AssetType, 
                            sources: Optional[List[str]] = None,
                            lookback_days: int = 7,
                            use_llm: bool = True) -> SentimentAnalysisResponse:
        """
        Enhanced analysis method that prioritizes detailed insights
        
        Args:
            symbol: Asset symbol to analyze
            asset_type: Type of asset (STOCK or CRYPTO)
            sources: List of sentiment sources to use
            lookback_days: Number of days to look back
            use_llm: Enable LLM-powered analysis
            
        Returns:
            Sentiment analysis response with enhanced insights
        """
        if sources is None:
            # Include news sentiment for enhanced insights
            sources = ["news_sentiment", "fear_greed_index"]
        
        request = SentimentAnalysisRequest(
            asset_type=asset_type,
            symbol=symbol,
            sources=sources,
            lookback_days=lookback_days,
            use_llm=use_llm
        )
        
        return self.analyze(request)


# Convenience function for simple usage with enhanced features
def analyze_sentiment(symbol: str, 
                     asset_type: AssetType, 
                     sources: Optional[List[str]] = None,
                     lookback_days: int = 7,
                     use_llm: bool = True) -> SentimentAnalysisResponse:
    """
    Convenience function for enhanced sentiment analysis
    
    Args:
        symbol: Asset symbol to analyze
        asset_type: Type of asset (STOCK or CRYPTO)
        sources: List of sentiment sources to use (default: enhanced sources)
        lookback_days: Number of days to look back
        use_llm: Enable LLM-powered news analysis
        
    Returns:
        Sentiment analysis response with enhanced insights
    """
    if sources is None:
        # Default to enhanced sources including news sentiment
        sources = ["news_sentiment", "fear_greed_index"]
    
    manager = SentimentManager(enable_llm=use_llm)
    request = SentimentAnalysisRequest(
        asset_type=asset_type,
        symbol=symbol,
        sources=sources,
        lookback_days=lookback_days,
        use_llm=use_llm
    )
    
    return manager.analyze(request) 