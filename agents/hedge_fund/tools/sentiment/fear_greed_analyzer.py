"""
Fear & Greed Index Analyzer

Analyzes market fear and greed indicators:
- Crypto: Alternative.me Fear & Greed Index (free)
- Stocks: VIX volatility index via Yahoo Finance (free)
"""

import requests
import yfinance as yf
from typing import Dict, Any

from agents.hedge_fund.tools.sentiment.base import SentimentAnalyzer, SentimentResult, AssetType, interpret_sentiment_score

class FearGreedAnalyzer(SentimentAnalyzer):
    """Analyzes market fear and greed indicators"""
    
    def __init__(self):
        super().__init__("fear_greed_index")
    
    def is_available(self) -> bool:
        """Fear & Greed analyzers are always available (no auth required)"""
        return True
    
    def analyze(self, symbol: str, asset_type: AssetType, **kwargs) -> SentimentResult:
        """
        Analyze fear/greed sentiment for the market
        
        Args:
            symbol: Asset symbol (used for metadata)
            asset_type: CRYPTO uses Alternative.me, STOCK uses VIX
        """
        try:
            if asset_type == AssetType.CRYPTO:
                return self._analyze_crypto_fear_greed()
            else:
                return self._analyze_stock_fear_greed()
                
        except Exception as e:
            return SentimentResult(
                source=self.name,
                error=f"Error in fear/greed analysis: {str(e)}"
            )
    
    def _analyze_crypto_fear_greed(self) -> SentimentResult:
        """Analyze crypto fear & greed using Alternative.me API"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"Alternative.me API returned status {response.status_code}")
            
            data = response.json()
            if not data or 'data' not in data or len(data['data']) == 0:
                raise Exception("No data returned from Alternative.me API")
            
            current_index = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            
            # Convert 0-100 scale to -1 to 1 scale
            sentiment_score = self._convert_fear_greed_to_sentiment(current_index)
            
            return SentimentResult(
                source=self.name,
                sentiment_score=sentiment_score,
                sentiment_signal=interpret_sentiment_score(sentiment_score, "Fear & Greed Index"),
                data_points=1,
                metadata={
                    "index_value": current_index,
                    "classification": classification,
                    "scale": "0-100 (Fear to Greed)",
                    "source": "Alternative.me",
                    "asset_type": "crypto"
                }
            )
            
        except Exception as e:
            return SentimentResult(
                source=self.name,
                error=f"Failed to fetch crypto Fear & Greed Index: {str(e)}"
            )
    
    def _analyze_stock_fear_greed(self) -> SentimentResult:
        """Analyze stock market fear using VIX"""
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5d")
            
            if vix_data.empty:
                raise Exception("No VIX data available")
            
            current_vix = round(float(vix_data['Close'].iloc[-1]), 2)
            
            # Convert VIX to sentiment score
            sentiment_score = self._convert_vix_to_sentiment(current_vix)
            
            # Calculate VIX trend
            vix_change = round(float(vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[0]), 2)
            vix_trend = "rising" if vix_change > 0 else "falling"
            
            return SentimentResult(
                source=self.name,
                sentiment_score=sentiment_score,
                sentiment_signal=interpret_sentiment_score(sentiment_score, "VIX Fear Index"),
                data_points=1,
                metadata={
                    "vix_value": current_vix,
                    "vix_change_5d": vix_change,
                    "vix_trend": vix_trend,
                    "interpretation": self._get_vix_interpretation(current_vix),
                    "source": "Yahoo Finance VIX",
                    "asset_type": "stock"
                }
            )
            
        except Exception as e:
            return SentimentResult(
                source=self.name,
                error=f"Error fetching VIX data: {str(e)}"
            )
    
    def _convert_fear_greed_to_sentiment(self, index_value: int) -> float:
        """Convert Fear & Greed Index (0-100) to sentiment score (-1 to 1)"""
        # 0-25: Very Bearish (-1 to -0.5)
        # 25-45: Bearish (-0.5 to -0.1)
        # 45-55: Neutral (-0.1 to 0.1)
        # 55-75: Bullish (0.1 to 0.5)
        # 75-100: Very Bullish (0.5 to 1.0)
        
        if index_value <= 25:
            return -1.0 + (index_value / 25) * 0.5
        elif index_value <= 45:
            return -0.5 + ((index_value - 25) / 20) * 0.4
        elif index_value <= 55:
            return -0.1 + ((index_value - 45) / 10) * 0.2
        elif index_value <= 75:
            return 0.1 + ((index_value - 55) / 20) * 0.4
        else:
            return 0.5 + ((index_value - 75) / 25) * 0.5
    
    def _convert_vix_to_sentiment(self, vix_value: float) -> float:
        """Convert VIX value to sentiment score (inverted - high VIX = fear = negative sentiment)"""
        # Typical VIX ranges: 10-20 (low fear), 20-30 (moderate), 30+ (high fear)
        if vix_value <= 15:
            return 0.5  # Low fear = bullish
        elif vix_value <= 20:
            return 0.2
        elif vix_value <= 25:
            return 0.0  # Neutral
        elif vix_value <= 30:
            return -0.3
        else:
            return -0.7  # High fear = bearish
    
    def _get_vix_interpretation(self, vix_value: float) -> str:
        """Get human-readable VIX interpretation"""
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
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        return {
            "crypto_source": "Alternative.me Fear & Greed Index",
            "stock_source": "VIX Volatility Index (Yahoo Finance)",
            "requires_api_key": False,
            "cost": "Free",
            "update_frequency": "Daily (crypto), Real-time (VIX)"
        } 