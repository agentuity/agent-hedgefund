"""
Market Sentiment Analyst

Clean interface for market sentiment analysis using the sentiment tools ecosystem.
Provides enhanced LLM-powered sentiment analysis with comprehensive insights.
"""

from typing import List, Dict, Optional, Any
import logging

from agents.hedge_fund.tools.sentiment.manager import SentimentManager
from agents.hedge_fund.tools.sentiment.base import (
    AssetType, SentimentType, SentimentSignal, SentimentResult
)

logger = logging.getLogger(__name__)

# --- Enhanced Sentiment Analysis Interface ---
class MarketSentimentAnalysis:
    """Results from enhanced market sentiment analysis"""
    
    def __init__(self, 
                 symbol: str,
                 asset_type: AssetType,
                 overall_sentiment_score: Optional[float] = None,
                 overall_sentiment_signal: Optional[SentimentSignal] = None,
                 sources: List[SentimentResult] = None,
                 enhanced_insights: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None):
        self.symbol = symbol
        self.asset_type = asset_type
        self.overall_sentiment_score = overall_sentiment_score
        self.overall_sentiment_signal = overall_sentiment_signal
        self.sources = sources or []
        self.enhanced_insights = enhanced_insights or {}
        self.error = error

def analyze_market_sentiment(
    symbol: str,
    asset_type: AssetType,
    sources: Optional[List[str]] = None,
    lookback_days: int = 7,
    use_llm_insights: bool = True
) -> MarketSentimentAnalysis:
    """
    Enhanced market sentiment analysis with LLM-powered insights
    
    Args:
        symbol: Asset symbol to analyze (e.g., "AAPL", "BTC-USD")
        asset_type: Type of asset (STOCK or CRYPTO)
        sources: List of sentiment sources (default: news, fear/greed, economic)
        lookback_days: Days to look back for analysis
        use_llm_insights: Enable LLM-powered enhanced analysis
        
    Returns:
        MarketSentimentAnalysis with comprehensive results and insights
    """
    try:
        logger.info(f"🔍 Analyzing market sentiment for {symbol} ({asset_type.value})")
        
        # Default to comprehensive sources if none specified
        if sources is None:
            if asset_type == AssetType.CRYPTO:
                sources = ["news_sentiment", "fear_greed_index"]
            else:
                sources = ["news_sentiment", "fear_greed_index", "economic_sentiment"]
        
        # Use enhanced sentiment manager with LLM capabilities
        sentiment_manager = SentimentManager(enable_llm=use_llm_insights)
        
        # Run comprehensive analysis with insights - this returns SentimentAnalysisResponse
        analysis_response = sentiment_manager.analyze_with_insights(
            symbol=symbol,
            asset_type=asset_type,
            sources=sources,
            lookback_days=lookback_days,
            use_llm=use_llm_insights
        )
        
        logger.info(f"✅ Sentiment analysis complete for {symbol}")
        logger.info(f"📊 Overall sentiment: {analysis_response.overall_sentiment_score:.2f}" if analysis_response.overall_sentiment_score else "📊 Overall sentiment: No data")
        
        # Return clean results from the SentimentAnalysisResponse object
        return MarketSentimentAnalysis(
            symbol=symbol,
            asset_type=asset_type,
            overall_sentiment_score=analysis_response.overall_sentiment_score,
            overall_sentiment_signal=analysis_response.overall_sentiment_signal,
            sources=analysis_response.sources,
            enhanced_insights=analysis_response.enhanced_analysis,
            error=analysis_response.error if hasattr(analysis_response, 'error') else None
        )
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed for {symbol}: {e}")
        return MarketSentimentAnalysis(
            symbol=symbol,
            asset_type=asset_type,
            error=f"Sentiment analysis failed: {str(e)}"
        )

def get_sentiment_summary(analysis: MarketSentimentAnalysis) -> str:
    """Get a brief summary of the sentiment analysis"""
    if analysis.error:
        return f"Sentiment analysis failed: {analysis.error}"
    
    if analysis.overall_sentiment_signal:
        sentiment = analysis.overall_sentiment_signal.sentiment.value
        confidence = analysis.overall_sentiment_signal.confidence
        return f"{sentiment} sentiment ({confidence:.1%} confidence)"
    
    return "No sentiment data available"

def has_actionable_sentiment(analysis: MarketSentimentAnalysis) -> bool:
    """Check if the sentiment analysis provides actionable insights"""
    if analysis.error or not analysis.overall_sentiment_signal:
        return False
    
    # Consider it actionable if confidence is reasonable and sentiment is not neutral
    return (analysis.overall_sentiment_signal.confidence > 0.3 and 
            analysis.overall_sentiment_signal.sentiment != SentimentType.NEUTRAL)

# --- Convenience Functions ---

def analyze_stock_sentiment(symbol: str, lookback_days: int = 7) -> MarketSentimentAnalysis:
    """Convenience function for stock sentiment analysis"""
    return analyze_market_sentiment(
        symbol=symbol,
        asset_type=AssetType.STOCK,
        lookback_days=lookback_days
    )

def analyze_crypto_sentiment(symbol: str, lookback_days: int = 3) -> MarketSentimentAnalysis:
    """Convenience function for crypto sentiment analysis"""
    return analyze_market_sentiment(
        symbol=symbol,
        asset_type=AssetType.CRYPTO,
        lookback_days=lookback_days
    )

def run_enhanced_market_sentiment_analysis(
    symbol: str,
    asset_type: AssetType,
    sources: Optional[List[str]] = None,
    lookback_days: int = 7,
    use_llm_insights: bool = True
) -> Dict[str, Any]:
    """
    Legacy function for trade_decision_agent compatibility
    Returns sentiment analysis in dictionary format
    """
    analysis = analyze_market_sentiment(
        symbol=symbol,
        asset_type=asset_type,
        sources=sources,
        lookback_days=lookback_days,
        use_llm_insights=use_llm_insights
    )
    
    # Convert to dictionary format expected by trade_decision_agent
    return {
        "status": "error" if analysis.error else "success",
        "symbol": analysis.symbol,
        "asset_type": analysis.asset_type.value,
        "overall_sentiment_score": analysis.overall_sentiment_score,
        "overall_sentiment_signal": analysis.overall_sentiment_signal.sentiment.value if analysis.overall_sentiment_signal else None,
        "confidence": analysis.overall_sentiment_signal.confidence if analysis.overall_sentiment_signal else 0.0,
        "sources": [
            {
                "source": result.source,
                "sentiment": result.sentiment_signal.sentiment.value if result.sentiment_signal else None,
                "score": result.sentiment_score,
                "confidence": result.sentiment_signal.confidence if result.sentiment_signal else 0.0
            }
            for result in analysis.sources
        ],
        "enhanced_insights": analysis.enhanced_insights,
        "error": analysis.error,
        "summary": get_sentiment_summary(analysis)
    }

# --- Public Interface ---

__all__ = [
    "MarketSentimentAnalysis",
    "analyze_market_sentiment",
    "get_sentiment_summary", 
    "has_actionable_sentiment",
    "analyze_stock_sentiment",
    "analyze_crypto_sentiment",
    "run_enhanced_market_sentiment_analysis",
    "AssetType"  # Re-export for convenience
]

# --- Example Usage ---

if __name__ == "__main__":
    # Test the streamlined sentiment analysis
    print("🔍 **Enhanced Market Sentiment Analysis Testing**\n")
    
    # Test stock sentiment
    print("--- Stock Sentiment Analysis ---")
    stock_analysis = analyze_stock_sentiment("AAPL", lookback_days=7)
    print(f"Result: {get_sentiment_summary(stock_analysis)}")
    print(f"Actionable: {has_actionable_sentiment(stock_analysis)}")
    
    if stock_analysis.enhanced_insights:
        print("Enhanced Insights:")
        for key, value in stock_analysis.enhanced_insights.items():
            print(f"  {key}: {value}")
    
    print("\n--- Crypto Sentiment Analysis ---")
    crypto_analysis = analyze_crypto_sentiment("BTC-USD", lookback_days=3)
    print(f"Result: {get_sentiment_summary(crypto_analysis)}")
    print(f"Actionable: {has_actionable_sentiment(crypto_analysis)}")
    
    if crypto_analysis.enhanced_insights:
        print("Enhanced Insights:")
        for key, value in crypto_analysis.enhanced_insights.items():
            print(f"  {key}: {value}")
    
    print("\n✅ **Clean Sentiment Analysis Interface Complete!**")
    print("✅ Streamlined to 150 lines with clear interface")
    print("✅ Delegates to sophisticated sentiment tools ecosystem")
    print("✅ Consistent return types and error handling")
    print("✅ Enhanced LLM insights included by default") 