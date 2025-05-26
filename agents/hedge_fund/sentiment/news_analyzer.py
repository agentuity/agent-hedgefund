"""
News Sentiment Analyzer

Analyzes sentiment from financial news articles using NewsAPI.
Requires NEWSAPI_KEY environment variable (free tier: 1,000 requests/day).

TODO: Allow a smart news searching agent to be used to find the most relevant news articles.
Use LLM to allow better search queries depending on what the LLM thinks is relevant.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
from textblob import TextBlob

from base import SentimentAnalyzer, SentimentResult, AssetType, interpret_sentiment_score


class NewsSentimentAnalyzer(SentimentAnalyzer):
    """Analyzes sentiment from financial news articles"""
    
    def __init__(self, max_articles: int = 50):
        super().__init__("news_sentiment")
        self.max_articles = max_articles
        self.api_key = os.getenv('NEWSAPI_KEY')
        
        # Company name mapping for better search results
        self.company_mapping = {
            'AAPL': 'Apple Inc',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'META': 'Meta Facebook',
            'NVDA': 'NVIDIA',
            'NFLX': 'Netflix',
            'AMD': 'Advanced Micro Devices',
            'INTC': 'Intel',
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'ADA': 'Cardano',
            'SOL': 'Solana',
            'DOGE': 'Dogecoin'
        }
        
        # Source credibility weights
        self.source_weights = {
            'reuters': 1.2,
            'bloomberg': 1.2,
            'wsj': 1.2,
            'financial times': 1.2,
            'cnbc': 1.1,
            'yahoo': 1.0,
            'marketwatch': 1.0,
            'seeking alpha': 1.0,
            'cnn': 1.0,
            'motley fool': 0.8,
            'zacks': 0.8,
            'benzinga': 0.8
        }
    
    def is_available(self) -> bool:
        """Check if NewsAPI key is available"""
        return self.api_key is not None
    
    def get_required_config(self) -> List[str]:
        """Return required configuration"""
        return ['NEWSAPI_KEY']
    
    def analyze(self, symbol: str, asset_type: AssetType, **kwargs) -> SentimentResult:
        """
        Analyze news sentiment for a symbol
        
        Args:
            symbol: Asset symbol to search for
            asset_type: Type of asset (used for metadata)
            lookback_days: Number of days to look back (default: 7)
        """
        if not self.is_available():
            return SentimentResult(
                source=self.name,
                error="NEWSAPI_KEY environment variable not set. Get free key from https://newsapi.org/"
            )
        
        lookback_days = kwargs.get('lookback_days', 7)
        
        try:
            articles = self._fetch_news_articles(symbol, lookback_days)
            
            if not articles:
                return SentimentResult(
                    source=self.name,
                    sentiment_score=0.0,
                    sentiment_signal=interpret_sentiment_score(0.0, "news"),
                    data_points=0,
                    metadata={"message": "No news articles found"}
                )
            
            sentiment_scores = self._analyze_articles_sentiment(articles, lookback_days)
            
            if not sentiment_scores:
                return SentimentResult(
                    source=self.name,
                    sentiment_score=0.0,
                    sentiment_signal=interpret_sentiment_score(0.0, "news"),
                    data_points=0,
                    metadata={"message": "No valid articles could be processed"}
                )
            
            avg_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 3)
            
            return SentimentResult(
                source=self.name,
                sentiment_score=avg_sentiment,
                sentiment_signal=interpret_sentiment_score(avg_sentiment, "news"),
                data_points=len(sentiment_scores),
                metadata={
                    "total_articles": len(articles),
                    "processed_articles": len(sentiment_scores),
                    "avg_sentiment": avg_sentiment,
                    "sentiment_range": [round(min(sentiment_scores), 3), round(max(sentiment_scores), 3)] if sentiment_scores else [0, 0],
                    "search_query": self._build_search_query(symbol)
                }
            )
            
        except Exception as e:
            return SentimentResult(
                source=self.name,
                error=f"Error in news sentiment analysis: {str(e)}"
            )
    
    def _fetch_news_articles(self, symbol: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Fetch news articles from NewsAPI"""
        query = self._build_search_query(symbol)
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': self.max_articles,
            'apiKey': self.api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"NewsAPI error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data.get('articles', [])
    
    def _build_search_query(self, symbol: str) -> str:
        """Build search query combining symbol and company name"""
        company_name = self.company_mapping.get(symbol.upper(), symbol)
        return f'"{symbol}" OR "{company_name}"'
    
    def _analyze_articles_sentiment(self, articles: List[Dict[str, Any]], lookback_days: int) -> List[float]:
        """Analyze sentiment of news articles"""
        sentiment_scores = []
        
        for article in articles:
            try:
                # Combine title and description for better context
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                if len(text.strip()) < 10:  # Skip very short texts
                    continue
                
                # TextBlob sentiment analysis
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity  # -1 to 1
                
                # Apply weighting factors
                weighted_score = self._apply_news_weights(sentiment_score, article, lookback_days)
                sentiment_scores.append(weighted_score)
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        return sentiment_scores
    
    def _apply_news_weights(self, sentiment_score: float, article: Dict[str, Any], lookback_days: int) -> float:
        """Apply news-specific weighting factors"""
        # Source credibility weighting
        source_name = article.get('source', {}).get('name', '').lower()
        source_weight = self._get_source_weight(source_name)
        
        # Recency weighting (newer articles matter more)
        recency_weight = self._get_recency_weight(article, lookback_days)
        
        return sentiment_score * source_weight * recency_weight
    
    def _get_source_weight(self, source_name: str) -> float:
        """Get credibility weight for news source"""
        if not source_name:
            return 1.0
        
        for source, weight in self.source_weights.items():
            if source in source_name:
                return weight
        
        return 0.9  # Default weight for unknown sources
    
    def _get_recency_weight(self, article: Dict[str, Any], lookback_days: int) -> float:
        """Calculate recency weight (newer articles get higher weight)"""
        try:
            pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
            days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
            return max(0.1, 1.0 - (days_old / lookback_days) * 0.5)
        except:
            return 1.0  # Default weight if date parsing fails
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        return {
            "max_articles": self.max_articles,
            "company_mapping_count": len(self.company_mapping),
            "source_weights_count": len(self.source_weights),
            "requires_api_key": True,
            "api_key_set": self.is_available(),
            "cost": "Free tier: 1,000 requests/day",
            "signup_url": "https://newsapi.org/"
        } 