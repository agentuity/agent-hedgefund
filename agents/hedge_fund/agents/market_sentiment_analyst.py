from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import requests
from datetime import datetime, timedelta
import json
import yfinance as yf
import os
from textblob import TextBlob
import time

from agents.hedge_fund.tools.sentiment.base import (
    AssetType,
    SentimentType,
    SentimentSignal,
)
class SentimentSignal(BaseModel):
    sentiment: SentimentType
    confidence: float = Field(..., description="Confidence level from 0.0 to 1.0")
    reason: str = Field(..., description="Human-readable explanation of the sentiment")

class SentimentSourceSpec(BaseModel):
    name: str
    params: Optional[Dict[str, Any]] = None

class MarketSentimentRequest(BaseModel):
    asset_type: AssetType
    symbol: str
    sources: List[SentimentSourceSpec]
    lookback_days: int = Field(default=7, description="Number of days to look back for sentiment data")

class SentimentResult(BaseModel):
    source: str
    sentiment_score: Optional[float] = None  # -1.0 (very bearish) to 1.0 (very bullish)
    sentiment_signal: Optional[SentimentSignal] = None
    data_points: Optional[int] = None  # Number of data points analyzed
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Additional source-specific data

class MarketSentimentToolOutput(BaseModel):
    asset_type: AssetType
    symbol: str
    lookback_days: int
    overall_sentiment_score: Optional[float] = None  # Aggregated sentiment score
    overall_sentiment_signal: Optional[SentimentSignal] = None
    sources: List[SentimentResult]
    data_fetch_error: Optional[str] = None

# --- Free API Implementations ---

def analyze_news_sentiment(symbol: str, lookback_days: int = 7) -> SentimentResult:
    """
    Real news sentiment using NewsAPI free tier (1,000 requests/day)
    Requires NEWSAPI_KEY environment variable
    """
    try:
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            return SentimentResult(
                source="news_sentiment",
                error="NEWSAPI_KEY environment variable not set. Get free key from https://newsapi.org/"
            )
        
        company_names = get_company_name_mapping(symbol.upper())
        
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        query = f'"{symbol}" OR "{company_names.get(symbol, symbol)}"'
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 50,  # Max for free tier
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return SentimentResult(
                source="news_sentiment",
                error=f"NewsAPI error: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        articles = data.get('articles', [])
        
        if not articles:
            return SentimentResult(
                source="news_sentiment",
                sentiment_score=0.0,
                sentiment_signal=interpret_sentiment_score(0.0, "news"),
                data_points=0,
                metadata={"message": "No news articles found"}
            )
        
        # Analyze sentiment using TextBlob (free)
        sentiment_scores = []
        processed_articles = 0
        
        for article in articles:
            try:
                # Combine title and description for better context
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                if len(text.strip()) < 10:  # Skip very short texts
                    continue
                
                # TextBlob sentiment analysis
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity  # -1 to 1
                
                # Weight by source credibility (basic implementation)
                source_weight = get_source_credibility(article.get('source', {}).get('name', ''))
                
                # Apply recency weighting (newer articles matter more)
                pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                recency_weight = max(0.1, 1.0 - (days_old / lookback_days) * 0.5)
                
                final_score = sentiment_score * source_weight * recency_weight
                sentiment_scores.append(final_score)
                processed_articles += 1
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        if not sentiment_scores:
            return SentimentResult(
                source="news_sentiment",
                sentiment_score=0.0,
                sentiment_signal=interpret_sentiment_score(0.0, "news"),
                data_points=0,
                metadata={"message": "No valid articles could be processed"}
            )
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        return SentimentResult(
            source="news_sentiment",
            sentiment_score=avg_sentiment,
            sentiment_signal=interpret_sentiment_score(avg_sentiment, "news"),
            data_points=processed_articles,
            metadata={
                "total_articles": len(articles),
                "processed_articles": processed_articles,
                "avg_sentiment": avg_sentiment,
                "sentiment_range": [min(sentiment_scores), max(sentiment_scores)]
            }
        )
        
    except Exception as e:
        return SentimentResult(
            source="news_sentiment",
            error=f"Error in news sentiment analysis: {str(e)}"
        )



def analyze_fear_greed_index(asset_type: AssetType) -> SentimentResult:
    """
    Real Fear & Greed Index using free APIs
    - Crypto: Alternative.me (completely free)
    - Stocks: VIX via Yahoo Finance (free)
    """
    try:
        if asset_type == AssetType.CRYPTO:
            # Alternative.me Fear & Greed Index (completely free)
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and 'data' in data and len(data['data']) > 0:
                    current_index = int(data['data'][0]['value'])
                    classification = data['data'][0]['value_classification']
                    
                    # Convert 0-100 scale to -1 to 1 scale
                    if current_index <= 25:
                        sentiment_score = -1.0 + (current_index / 25) * 0.5
                    elif current_index <= 45:
                        sentiment_score = -0.5 + ((current_index - 25) / 20) * 0.4
                    elif current_index <= 55:
                        sentiment_score = -0.1 + ((current_index - 45) / 10) * 0.2
                    elif current_index <= 75:
                        sentiment_score = 0.1 + ((current_index - 55) / 20) * 0.4
                    else:
                        sentiment_score = 0.5 + ((current_index - 75) / 25) * 0.5
                    
                    return SentimentResult(
                        source="fear_greed_index",
                        sentiment_score=sentiment_score,
                        sentiment_signal=interpret_sentiment_score(sentiment_score, "Fear & Greed Index"),
                        data_points=1,
                        metadata={
                            "index_value": current_index,
                            "classification": classification,
                            "scale": "0-100 (Fear to Greed)",
                            "source": "Alternative.me"
                        }
                    )
            
            return SentimentResult(
                source="fear_greed_index",
                error="Failed to fetch crypto Fear & Greed Index"
            )
            
        else:  # STOCK
            # Use VIX from Yahoo Finance (free)
            try:
                vix_ticker = yf.Ticker("^VIX")
                vix_data = vix_ticker.history(period="5d")
                
                if vix_data.empty:
                    raise Exception("No VIX data available")
                
                current_vix = float(vix_data['Close'].iloc[-1])
                
                # VIX interpretation (inverted - high VIX = fear = negative sentiment)
                # Typical VIX ranges: 10-20 (low fear), 20-30 (moderate), 30+ (high fear)
                if current_vix <= 15:
                    sentiment_score = 0.5  # Low fear = bullish
                elif current_vix <= 20:
                    sentiment_score = 0.2
                elif current_vix <= 25:
                    sentiment_score = 0.0  # Neutral
                elif current_vix <= 30:
                    sentiment_score = -0.3
                else:
                    sentiment_score = -0.7  # High fear = bearish
                
                # Get VIX trend (5-day change)
                vix_change = float(vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[0])
                vix_trend = "rising" if vix_change > 0 else "falling"
                
                return SentimentResult(
                    source="fear_greed_index",
                    sentiment_score=sentiment_score,
                    sentiment_signal=interpret_sentiment_score(sentiment_score, "VIX Fear Index"),
                    data_points=1,
                    metadata={
                        "vix_value": current_vix,
                        "vix_change_5d": vix_change,
                        "vix_trend": vix_trend,
                        "interpretation": get_vix_interpretation(current_vix),
                        "source": "Yahoo Finance VIX"
                    }
                )
                
            except Exception as e:
                return SentimentResult(
                    source="fear_greed_index",
                    error=f"Error fetching VIX data: {str(e)}"
                )
            
    except Exception as e:
        return SentimentResult(
            source="fear_greed_index",
            error=f"Error in fear/greed analysis: {str(e)}"
        )

def analyze_economic_sentiment(asset_type: AssetType) -> SentimentResult:
    """
    Real economic sentiment using FRED API (completely free)
    Federal Reserve Economic Data
    """
    try:
        fred_api_key = os.getenv('FRED_API_KEY')
        if not fred_api_key:
            return SentimentResult(
                source="economic_sentiment",
                error="FRED_API_KEY environment variable not set. Get free key from https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        
        # Key economic indicators
        indicators = {
            'unemployment': 'UNRATE',
            'inflation': 'CPIAUCSL',
            'gdp_growth': 'GDP',
            'consumer_confidence': 'UMCSENT',
            'yield_10y': 'GS10',
            'yield_2y': 'GS2'
        }
        
        economic_data = {}
        
        for name, series_id in indicators.items():
            try:
                url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': fred_api_key,
                    'file_type': 'json',
                    'limit': 2,  # Get last 2 data points for trend
                    'sort_order': 'desc'
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    observations = data.get('observations', [])
                    
                    if len(observations) >= 1:
                        current_value = float(observations[0]['value'])
                        economic_data[name] = {
                            'current': current_value,
                            'date': observations[0]['date']
                        }
                        
                        # Calculate trend if we have 2 data points
                        if len(observations) >= 2:
                            prev_value = float(observations[1]['value'])
                            economic_data[name]['change'] = current_value - prev_value
                            economic_data[name]['trend'] = 'improving' if current_value > prev_value else 'declining'
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                continue
        
        if not economic_data:
            return SentimentResult(
                source="economic_sentiment",
                error="No economic data could be fetched"
            )
        
        # Calculate economic sentiment score
        sentiment_factors = []
        
        # Unemployment (lower is better)
        if 'unemployment' in economic_data:
            unemployment = economic_data['unemployment']['current']
            if unemployment < 4.0:
                sentiment_factors.append(0.3)  # Very good
            elif unemployment < 6.0:
                sentiment_factors.append(0.1)  # Good
            elif unemployment < 8.0:
                sentiment_factors.append(-0.1)  # Concerning
            else:
                sentiment_factors.append(-0.4)  # Bad
        
        # Consumer confidence (higher is better)
        if 'consumer_confidence' in economic_data:
            confidence = economic_data['consumer_confidence']['current']
            if confidence > 100:
                sentiment_factors.append(0.2)
            elif confidence > 80:
                sentiment_factors.append(0.1)
            elif confidence > 60:
                sentiment_factors.append(-0.1)
            else:
                sentiment_factors.append(-0.3)
        
        # Yield curve (10Y - 2Y spread)
        if 'yield_10y' in economic_data and 'yield_2y' in economic_data:
            yield_spread = economic_data['yield_10y']['current'] - economic_data['yield_2y']['current']
            if yield_spread > 1.0:
                sentiment_factors.append(0.2)  # Normal curve
            elif yield_spread > 0:
                sentiment_factors.append(0.0)  # Flattening
            else:
                sentiment_factors.append(-0.4)  # Inverted (recession signal)
        
        if not sentiment_factors:
            avg_sentiment = 0.0
        else:
            avg_sentiment = sum(sentiment_factors) / len(sentiment_factors)
        
        return SentimentResult(
            source="economic_sentiment",
            sentiment_score=avg_sentiment,
            sentiment_signal=interpret_sentiment_score(avg_sentiment, "economic indicators"),
            data_points=len(economic_data),
            metadata={
                "indicators": economic_data,
                "sentiment_factors": sentiment_factors,
                "data_source": "Federal Reserve Economic Data (FRED)"
            }
        )
        
    except Exception as e:
        return SentimentResult(
            source="economic_sentiment",
            error=f"Error in economic sentiment analysis: {str(e)}"
        )

# --- Helper Functions ---

def get_company_name_mapping(symbol: str) -> Dict[str, str]:
    """Map stock symbols to company names for better news search"""
    mapping = {
        'AAPL': 'Apple Inc',
        'GOOGL': 'Google Alphabet',
        'MSFT': 'Microsoft',
        'TSLA': 'Tesla',
        'AMZN': 'Amazon',
        'META': 'Meta Facebook',
        'NVDA': 'NVIDIA',
        'NFLX': 'Netflix',
        'AMD': 'Advanced Micro Devices',
        'INTC': 'Intel'
    }
    return mapping

def get_source_credibility(source_name: str) -> float:
    """Basic source credibility weighting"""
    if not source_name:
        return 1.0
    
    source_name = source_name.lower()
    
    # High credibility sources
    if any(name in source_name for name in ['reuters', 'bloomberg', 'wsj', 'financial times', 'cnbc']):
        return 1.2
    
    # Medium credibility
    elif any(name in source_name for name in ['yahoo', 'marketwatch', 'seeking alpha', 'cnn']):
        return 1.0
    
    # Lower credibility
    elif any(name in source_name for name in ['motley fool', 'zacks', 'benzinga']):
        return 0.8
    
    # Unknown sources
    else:
        return 0.9

def get_vix_interpretation(vix_value: float) -> str:
    """Interpret VIX values"""
    if vix_value < 12:
        return "Very Low Fear (Complacency Risk)"
    elif vix_value < 20:
        return "Low Fear (Normal Market)"
    elif vix_value < 30:
        return "Elevated Fear (Market Stress)"
    elif vix_value < 40:
        return "High Fear (Significant Volatility)"
    else:
        return "Extreme Fear (Market Panic)"

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

def aggregate_sentiment_scores(sentiment_results: List[SentimentResult]) -> tuple[float, SentimentSignal]:
    """Aggregate multiple sentiment scores into overall sentiment"""
    
    valid_results = [r for r in sentiment_results if r.sentiment_score is not None and r.error is None]
    
    if not valid_results:
        return 0.0, SentimentSignal(
            sentiment=SentimentType.NEUTRAL,
            confidence=0.0,
            reason="No valid sentiment data available for aggregation"
        )
    
    # Weight different sources
    source_weights = {
        "news_sentiment": 0.40,
        "fear_greed_index": 0.30,
        "economic_sentiment": 0.30
    }
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for result in valid_results:
        weight = source_weights.get(result.source, 0.15)
        weighted_sum += result.sentiment_score * weight
        total_weight += weight
    
    if total_weight == 0:
        overall_score = 0.0
    else:
        overall_score = weighted_sum / total_weight
    
    # Calculate confidence based on agreement between sources
    scores = [r.sentiment_score for r in valid_results]
    score_variance = sum((s - overall_score) ** 2 for s in scores) / len(scores)
    agreement_confidence = max(0.0, 1.0 - score_variance * 2)
    
    overall_signal = interpret_sentiment_score(overall_score, "aggregated")
    overall_signal.confidence = min(overall_signal.confidence * agreement_confidence, 1.0)
    
    return overall_score, overall_signal

# --- Core Tool Function ---

def run_market_sentiment_tool(request: MarketSentimentRequest) -> MarketSentimentToolOutput:
    """Main function to run market sentiment analysis using free APIs"""
    
    sentiment_results: List[SentimentResult] = []
    data_fetch_error: Optional[str] = None
    
    try:
        for source_spec in request.sources:
            result = None
            
            try:
                if source_spec.name.lower() == "news_sentiment":
                    result = analyze_news_sentiment(request.symbol, request.lookback_days)
                elif source_spec.name.lower() == "fear_greed_index":
                    result = analyze_fear_greed_index(request.asset_type)
                elif source_spec.name.lower() == "economic_sentiment":
                    result = analyze_economic_sentiment(request.asset_type)
                else:
                    result = SentimentResult(
                        source=source_spec.name,
                        error=f"Unknown sentiment source: {source_spec.name}"
                    )
                
                if result:
                    sentiment_results.append(result)
                    
            except Exception as e:
                sentiment_results.append(SentimentResult(
                    source=source_spec.name,
                    error=f"Error processing {source_spec.name}: {str(e)}"
                ))
        
        # Aggregate overall sentiment
        overall_score, overall_signal = aggregate_sentiment_scores(sentiment_results)
        
        return MarketSentimentToolOutput(
            asset_type=request.asset_type,
            symbol=request.symbol,
            lookback_days=request.lookback_days,
            overall_sentiment_score=overall_score,
            overall_sentiment_signal=overall_signal,
            sources=sentiment_results,
            data_fetch_error=data_fetch_error
        )
        
    except Exception as e:
        data_fetch_error = f"Critical error in sentiment analysis: {str(e)}"
        return MarketSentimentToolOutput(
            asset_type=request.asset_type,
            symbol=request.symbol,
            lookback_days=request.lookback_days,
            overall_sentiment_score=None,
            overall_sentiment_signal=None,
            sources=sentiment_results,
            data_fetch_error=data_fetch_error
        )

# --- Example Usage ---

if __name__ == "__main__":
    print("=== Free API Market Sentiment Analysis ===")
    print("Required environment variables:")
    print("- NEWSAPI_KEY: Get free at https://newsapi.org/")
    print("- FRED_API_KEY: Get free at https://fred.stlouisfed.org/docs/api/api_key.html")
    print()
    
    # Example Stock Sentiment Analysis
    stock_req = MarketSentimentRequest(
        asset_type=AssetType.STOCK,
        symbol="AAPL",
        lookback_days=7,
        sources=[
            SentimentSourceSpec(name="news_sentiment"),
            SentimentSourceSpec(name="fear_greed_index"),
            SentimentSourceSpec(name="economic_sentiment")
        ]
    )
    
    print("--- Stock Market Sentiment Analysis (Free APIs) ---")
    stock_result = run_market_sentiment_tool(stock_req)
    print(stock_result.model_dump_json(indent=2))
    print("\n")
    
    # Example Crypto Sentiment Analysis
    crypto_req = MarketSentimentRequest(
        asset_type=AssetType.CRYPTO,
        symbol="BTC-USD",
        lookback_days=3,
        sources=[
            SentimentSourceSpec(name="news_sentiment"),
            SentimentSourceSpec(name="fear_greed_index"),
            SentimentSourceSpec(name="economic_sentiment")
        ]
    )
    
    print("--- Crypto Market Sentiment Analysis (Free APIs) ---")
    crypto_result = run_market_sentiment_tool(crypto_req)
    print(crypto_result.model_dump_json(indent=2)) 