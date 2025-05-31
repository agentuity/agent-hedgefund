"""
Enhanced News Sentiment Analyzer

Analyzes sentiment from financial news articles using NewsAPI.
Enhanced with LLM-powered interpretation for actionable insights.
Requires NEWSAPI_KEY environment variable (free tier: 1,000 requests/day).
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from textblob import TextBlob
import logging


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from agents.hedge_fund.tools.sentiment.base import SentimentAnalyzer, SentimentResult, AssetType, interpret_sentiment_score, SentimentSignal, SentimentType

logger = logging.getLogger(__name__)

class NewsInsight(BaseModel):
    """Structured news insight from LLM analysis"""
    key_themes: List[str] = Field(description="3-5 key themes from the news")
    market_moving_events: List[str] = Field(description="Specific events that could impact price")
    sentiment_reasoning: str = Field(description="Why the sentiment is bullish/bearish/neutral")
    risk_factors: List[str] = Field(description="Potential risks mentioned in news")
    opportunities: List[str] = Field(description="Potential opportunities mentioned")
    overall_sentiment: str = Field(description="VERY_BULLISH, BULLISH, NEUTRAL, BEARISH, or VERY_BEARISH")
    confidence_score: float = Field(description="Confidence in analysis from 0.0 to 1.0")


class NewsSentimentAnalyzer(SentimentAnalyzer):
    """Enhanced news sentiment analyzer with LLM interpretation"""
    
    def __init__(self, max_articles: int = 30, use_llm: bool = True):
        super().__init__("news_sentiment")
        self.max_articles = max_articles
        self.use_llm = use_llm
        self.api_key = os.getenv('NEWSAPI_KEY')
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize LLM if available
        self.llm = None
        if self.use_llm and os.getenv('OPENAI_API_KEY'):
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=1.0,
                    max_tokens=1000
                )
                self.logger.info("LLM (GPT-4o-mini) initialized successfully for enhanced news analysis")
            except Exception as e:
                self.logger.warning("Could not initialize LLM for enhanced analysis: %s", e)
                self.use_llm = False
        elif self.use_llm:
            self.logger.info("LLM requested but OPENAI_API_KEY not found, falling back to basic analysis")
        
        self.company_mapping = {
            'AAPL': 'Apple Inc',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft Corporation',
            'TSLA': 'Tesla Inc',
            'AMZN': 'Amazon.com Inc',
            'META': 'Meta Platforms Facebook',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc',
            'AMD': 'Advanced Micro Devices',
            'INTC': 'Intel Corporation',
            'JPM': 'JPMorgan Chase',
            'BAC': 'Bank of America',
            'BRK.B': 'Berkshire Hathaway',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc',
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'ADA': 'Cardano',
            'SOL': 'Solana',
            'DOGE': 'Dogecoin',
            'AVAX': 'Avalanche',
            'DOT': 'Polkadot'
        }
        
        # Source credibility weights
        self.source_weights = {
            'reuters': 1.3,
            'bloomberg': 1.3,
            'wsj': 1.3,
            'wall street journal': 1.3,
            'financial times': 1.3,
            'cnbc': 1.2,
            'marketwatch': 1.1,
            'yahoo finance': 1.1,
            'seeking alpha': 1.0,
            'cnn business': 1.0,
            'forbes': 1.0,
            'barrons': 1.1,
            'motley fool': 0.8,
            'zacks': 0.8,
            'benzinga': 0.8
        }
        
        self.logger.info("Enhanced News Sentiment Analyzer initialized - LLM enabled: %s", self.llm is not None)
    
    def is_available(self) -> bool:
        """Check if NewsAPI key is available"""
        return self.api_key is not None
    
    def get_required_config(self) -> List[str]:
        """Return required configuration"""
        config = ['NEWSAPI_KEY']
        if self.use_llm:
            config.append('OPENAI_API_KEY (optional for enhanced analysis)')
        return config
    
    def analyze(self, symbol: str, asset_type: AssetType, **kwargs) -> SentimentResult:
        """
        Analyze news sentiment for a symbol with enhanced LLM interpretation
        
        Args:
            symbol: Asset symbol to search for
            asset_type: Type of asset (used for metadata)
            lookback_days: Number of days to look back (default: 7)
            use_llm_override: Override LLM usage for this analysis
        """
        if not self.is_available():
            self.logger.error("NewsAPI key not configured - cannot perform news analysis")
            return SentimentResult(
                source=self.name,
                error="NEWSAPI_KEY environment variable not set. Get free key from https://newsapi.org/"
            )
        
        lookback_days = kwargs.get('lookback_days', 7)
        use_llm_override = kwargs.get('use_llm_override', self.use_llm)
        
        self.logger.info("Starting news sentiment analysis for %s (lookback: %d days, LLM: %s)", 
                        symbol, lookback_days, use_llm_override)
        
        try:
            # Fetch news articles
            articles = self._fetch_news_articles(symbol, lookback_days)
            
            if not articles:
                self.logger.warning("No news articles found for symbol %s", symbol)
                return SentimentResult(
                    source=self.name,
                    sentiment_score=0.0,
                    sentiment_signal=interpret_sentiment_score(0.0, "news"),
                    data_points=0,
                    metadata={"message": "No news articles found"}
                )
            
            self.logger.info("Fetched %d articles for %s, proceeding with analysis", len(articles), symbol)
            
            # Basic sentiment analysis with TextBlob
            basic_analysis = self._analyze_articles_sentiment_basic(articles, lookback_days)
            
            # Enhanced LLM analysis if available
            llm_insight = None
            if use_llm_override and self.llm and articles:
                try:
                    self.logger.info("Running enhanced LLM analysis on top %d articles", min(10, len(articles)))
                    llm_insight = self._analyze_with_llm(symbol, articles[:10])  # Analyze top 10 articles
                    if llm_insight:
                        self.logger.info("LLM analysis completed successfully with %d themes and %d events", 
                                       len(llm_insight.key_themes), len(llm_insight.market_moving_events))
                except Exception as e:
                    self.logger.warning("LLM analysis failed, falling back to basic analysis: %s", e)
            
            # Combine basic and LLM analysis
            final_result = self._combine_analysis_results(
                symbol, basic_analysis, llm_insight, articles
            )
            
            analysis_method = final_result.metadata.get("analysis_method", "unknown") if final_result.metadata else "unknown"
            self.logger.info("News sentiment analysis completed for %s: score=%.3f, method=%s", 
                           symbol, final_result.sentiment_score or 0.0, analysis_method)
            
            return final_result
            
        except Exception as e:
            self.logger.error("Error in news sentiment analysis for %s: %s", symbol, e)
            return SentimentResult(
                source=self.name,
                error=f"Error in news sentiment analysis: {str(e)}"
            )
    
    def _analyze_with_llm(self, symbol: str, articles: List[Dict[str, Any]]) -> Optional[NewsInsight]:
        """Use LLM to analyze news articles for deeper insights"""
        if not self.llm:
            return None
        
        # Prepare news content for LLM
        news_content = self._prepare_news_for_llm(articles)
        company_name = self.company_mapping.get(symbol.upper(), symbol)
        
        self.logger.debug("Prepared %d articles for LLM analysis", len(articles))
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_template("""
You are a professional financial analyst specializing in news sentiment analysis for trading decisions.

Analyze the following news articles about {company_name} ({symbol}) and provide actionable insights:

{news_content}

Please analyze these articles and provide:
1. Key themes affecting the stock/asset
2. Specific market-moving events or announcements
3. Clear reasoning for the overall sentiment
4. Risk factors that could impact price negatively
5. Opportunities that could drive price higher
6. Overall sentiment classification
7. Your confidence in this analysis

Respond in JSON format with the following structure:
{{
    "key_themes": ["theme1", "theme2", "theme3"],
    "market_moving_events": ["event1", "event2"],
    "sentiment_reasoning": "Clear explanation of why sentiment is bullish/bearish/neutral",
    "risk_factors": ["risk1", "risk2"],
    "opportunities": ["opportunity1", "opportunity2"],
    "overall_sentiment": "VERY_BULLISH|BULLISH|NEUTRAL|BEARISH|VERY_BEARISH",
    "confidence_score": 0.85
}}

Focus on actionable insights that would help a trader make informed decisions.
""")
        
        # Create output parser
        parser = JsonOutputParser(pydantic_object=NewsInsight)
        
        # Create chain
        chain = prompt | self.llm | parser
        
        try:
            # Invoke the chain
            result = chain.invoke({
                "company_name": company_name,
                "symbol": symbol,
                "news_content": news_content
            })
            
            insight = NewsInsight(**result)
            self.logger.debug("LLM analysis successful: sentiment=%s, confidence=%.2f", 
                            insight.overall_sentiment, insight.confidence_score)
            return insight
            
        except Exception as e:
            self.logger.error("LLM parsing error: %s", e)
            return None
    
    def _prepare_news_for_llm(self, articles: List[Dict[str, Any]]) -> str:
        """Prepare news articles for LLM analysis"""
        formatted_articles = []
        
        for i, article in enumerate(articles[:10], 1):  # Limit to top 10 articles
            title = article.get('title', 'No title')
            description = article.get('description', 'No description')
            source = article.get('source', {}).get('name', 'Unknown')
            pub_date = article.get('publishedAt', 'Unknown date')
            
            formatted_article = f"""
Article {i}:
Source: {source}
Date: {pub_date}
Title: {title}
Summary: {description}
---
"""
            formatted_articles.append(formatted_article)
        
        return "\n".join(formatted_articles)
    
    def _analyze_articles_sentiment_basic(self, articles: List[Dict[str, Any]], lookback_days: int) -> Dict[str, Any]:
        """Basic sentiment analysis using TextBlob"""
        sentiment_scores = []
        processed_articles = []
        
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
                
                processed_articles.append({
                    'title': article.get('title', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'sentiment': weighted_score,
                    'pub_date': article.get('publishedAt', '')
                })
                
            except Exception as e:
                self.logger.warning("Error processing article: %s", e)
                continue
        
        avg_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 3) if sentiment_scores else 0.0
        
        self.logger.debug("Basic sentiment analysis: processed %d/%d articles, avg sentiment: %.3f", 
                         len(sentiment_scores), len(articles), avg_sentiment)
        
        return {
            'avg_sentiment': avg_sentiment,
            'sentiment_scores': sentiment_scores,
            'processed_articles': processed_articles,
            'total_articles': len(articles)
        }
    
    def _combine_analysis_results(self, symbol: str, basic_analysis: Dict[str, Any], 
                                llm_insight: Optional[NewsInsight], articles: List[Dict[str, Any]]) -> SentimentResult:
        """Combine basic sentiment analysis with LLM insights"""
        
        # Start with basic sentiment
        base_sentiment = basic_analysis['avg_sentiment']
        
        # If we have LLM insight, blend the results
        if llm_insight:
            # Convert LLM sentiment to numeric
            llm_sentiment_map = {
                'VERY_BULLISH': 0.8,
                'BULLISH': 0.4,
                'NEUTRAL': 0.0,
                'BEARISH': -0.4,
                'VERY_BEARISH': -0.8
            }
            
            llm_sentiment_score = llm_sentiment_map.get(llm_insight.overall_sentiment, 0.0)
            
            # Weighted average (LLM gets more weight due to sophistication)
            final_sentiment = round((base_sentiment * 0.3 + llm_sentiment_score * 0.7), 3)
            
            # Create enhanced sentiment signal
            enhanced_signal = SentimentSignal(
                sentiment=SentimentType(llm_insight.overall_sentiment),
                confidence=llm_insight.confidence_score,
                reason=llm_insight.sentiment_reasoning
            )
            
            # Enhanced metadata with LLM insights
            metadata = {
                "total_articles": basic_analysis['total_articles'],
                "processed_articles": len(basic_analysis['sentiment_scores']),
                "basic_sentiment": base_sentiment,
                "llm_sentiment": llm_sentiment_score,
                "final_sentiment": final_sentiment,
                "key_themes": llm_insight.key_themes,
                "market_moving_events": llm_insight.market_moving_events,
                "risk_factors": llm_insight.risk_factors,
                "opportunities": llm_insight.opportunities,
                "llm_confidence": llm_insight.confidence_score,
                "analysis_method": "enhanced_llm"
            }
            
            self.logger.info("Combined analysis for %s: basic=%.3f, llm=%.3f, final=%.3f", 
                           symbol, base_sentiment, llm_sentiment_score, final_sentiment)
        else:
            # Fall back to basic analysis
            final_sentiment = base_sentiment
            enhanced_signal = interpret_sentiment_score(final_sentiment, "news")
            
            metadata = {
                "total_articles": basic_analysis['total_articles'],
                "processed_articles": len(basic_analysis['sentiment_scores']),
                "avg_sentiment": final_sentiment,
                "sentiment_range": [
                    round(min(basic_analysis['sentiment_scores']), 3), 
                    round(max(basic_analysis['sentiment_scores']), 3)
                ] if basic_analysis['sentiment_scores'] else [0, 0],
                "analysis_method": "basic_textblob"
            }
            
            self.logger.info("Basic analysis for %s: sentiment=%.3f (no LLM available)", symbol, final_sentiment)
        
        return SentimentResult(
            source=self.name,
            sentiment_score=final_sentiment,
            sentiment_signal=enhanced_signal,
            data_points=len(basic_analysis['sentiment_scores']),
            metadata=metadata
        )

    def _fetch_news_articles(self, symbol: str, lookback_days: int) -> List[Dict[str, Any]]:
        """Fetch news articles from NewsAPI with enhanced search queries"""
        query = self._build_enhanced_search_query(symbol)
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        self.logger.debug("Fetching news articles with query: %s, from_date: %s", query, from_date)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': self.max_articles,
            'apiKey': self.api_key,
            'domains': 'reuters.com,bloomberg.com,wsj.com,cnbc.com,marketwatch.com,yahoo.com'  # Focus on financial sources
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            self.logger.error("NewsAPI request failed: %d - %s", response.status_code, response.text)
            raise Exception(f"NewsAPI error: {response.status_code} - {response.text}")
        
        data = response.json()
        articles = data.get('articles', [])
        
        self.logger.debug("Successfully fetched %d articles from NewsAPI", len(articles))
        return articles
    
    def _build_enhanced_search_query(self, symbol: str) -> str:
        """Build enhanced search query with financial keywords"""
        company_name = self.company_mapping.get(symbol.upper(), symbol)
        
        # Create more targeted query
        base_query = f'"{symbol}" OR "{company_name}"'
        
        # Add financial context for better results
        if symbol.upper() in ['BTC', 'ETH', 'ADA', 'SOL', 'DOGE', 'AVAX', 'DOT']:
            # Crypto-specific keywords
            base_query += ' AND (cryptocurrency OR crypto OR bitcoin OR blockchain OR "digital asset")'
            self.logger.debug("Built crypto-enhanced search query for %s", symbol)
        else:
            # Stock-specific keywords
            base_query += ' AND (stock OR shares OR earnings OR revenue OR "quarterly results" OR SEC OR NYSE OR NASDAQ)'
            self.logger.debug("Built stock-enhanced search query for %s", symbol)
        
        return base_query
    
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
        except Exception as e:
            self.logger.debug("Could not parse article date: %s", e)
            return 1.0  # Default weight if date parsing fails
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        return {
            "max_articles": self.max_articles,
            "company_mapping_count": len(self.company_mapping),
            "source_weights_count": len(self.source_weights),
            "requires_api_key": True,
            "api_key_set": self.is_available(),
            "llm_available": self.llm is not None,
            "cost": "Free tier: 1,000 requests/day",
            "signup_url": "https://newsapi.org/",
            "enhanced_features": "LLM-powered insights, key themes, risk analysis"
        }