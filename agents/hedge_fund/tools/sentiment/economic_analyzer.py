"""
Economic Sentiment Analyzer

Analyzes macroeconomic sentiment using Federal Reserve Economic Data (FRED).
Requires FRED_API_KEY environment variable (completely free, unlimited requests).
"""

import os
import requests
import time
from typing import List, Dict, Any
from base import SentimentAnalyzer, SentimentResult, AssetType, interpret_sentiment_score


class EconomicSentimentAnalyzer(SentimentAnalyzer):
    """Analyzes macroeconomic sentiment using FRED data"""
    
    def __init__(self):
        super().__init__("economic_sentiment")
        self.api_key = os.getenv('FRED_API_KEY')
        
        # Economic indicators to track
        self.indicators = {
            'unemployment': 'UNRATE',
            'inflation': 'CPIAUCSL', 
            'gdp_growth': 'GDP',
            'consumer_confidence': 'UMCSENT',
            'yield_10y': 'GS10',
            'yield_2y': 'GS2'
        }
        
        # Asset-specific indicator sets
        self.crypto_indicators = [
            'regulatory_sentiment',
            'adoption_trends',
            'institutional_acceptance', 
            'monetary_policy',
            'risk_appetite'
        ]
        self.stock_indicators = [
            'fed_policy_sentiment',
            'economic_growth_outlook',
            'inflation_expectations',
            'earnings_season_sentiment',
            'geopolitical_stability'
        ]
    
    def is_available(self) -> bool:
        """Check if FRED API key is available"""
        return bool(self.api_key)
    
    def get_required_config(self) -> List[str]:
        """Return required configuration"""
        return ['FRED_API_KEY']
    
    def analyze(self, symbol: str, asset_type: AssetType, **kwargs) -> SentimentResult:
        """
        Analyze economic sentiment
        
        Args:
            symbol: Asset symbol (used for metadata)
            asset_type: Type of asset (affects which indicators are emphasized)
        """
        if not self.is_available():
            return SentimentResult(
                source=self.name,
                error="FRED_API_KEY environment variable not set. Get free key from https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        
        try:
            economic_data = self._fetch_economic_indicators()
            
            if not economic_data:
                return SentimentResult(
                    source=self.name,
                    error="No economic data could be fetched"
                )
            
            sentiment_score = round(self._calculate_economic_sentiment(economic_data, asset_type), 3)
            
            return SentimentResult(
                source=self.name,
                sentiment_score=sentiment_score,
                sentiment_signal=interpret_sentiment_score(sentiment_score, "economic indicators"),
                data_points=len(economic_data),
                metadata={
                    "indicators": economic_data,
                    "asset_type": asset_type.value,
                    "data_source": "Federal Reserve Economic Data (FRED)",
                    "relevant_factors": self._get_relevant_factors(asset_type)
                }
            )
            
        except Exception as e:
            return SentimentResult(
                source=self.name,
                error=f"Error in economic sentiment analysis: {str(e)}"
            )
     
    def _fetch_economic_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Fetch economic indicators from FRED API"""
        economic_data = {}
        
        for name, series_id in self.indicators.items():
            try:
                indicator_data = self._fetch_fred_series(series_id)
                if indicator_data:
                    economic_data[name] = indicator_data
                
                # Rate limiting (be nice to FRED)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                continue
        
        return economic_data
    
    def _fetch_fred_series(self, series_id: str) -> Dict[str, Any]:
        """Fetch a specific economic series from FRED"""
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'limit': 2,  # Get last 2 data points for trend calculation
            'sort_order': 'desc'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"FRED API returned status {response.status_code}")
        
        data = response.json()
        observations = data.get('observations', [])
        
        if len(observations) >= 1:
            current_value = float(observations[0]['value'])
            result = {
                'current': round(current_value, 3),
                'date': observations[0]['date']
            }
            
            # Calculate trend if we have 2 data points
            if len(observations) >= 2:
                prev_value = float(observations[1]['value'])
                change = current_value - prev_value
                result['change'] = round(change, 3)
                result['trend'] = 'improving' if change > 0 else 'declining'
            
            return result
        
        return None
    
    def _calculate_economic_sentiment(self, economic_data: Dict[str, Dict[str, Any]], asset_type: AssetType) -> float:
        """Calculate overall economic sentiment score"""
        sentiment_factors = []
        
        # Unemployment analysis (lower is better)
        if 'unemployment' in economic_data:
            unemployment = economic_data['unemployment']['current']
            unemployment_sentiment = self._analyze_unemployment(unemployment)
            sentiment_factors.append(unemployment_sentiment)
        
        # Consumer confidence analysis (higher is better)
        if 'consumer_confidence' in economic_data:
            confidence = economic_data['consumer_confidence']['current']
            confidence_sentiment = self._analyze_consumer_confidence(confidence)
            sentiment_factors.append(confidence_sentiment)
        
        # Yield curve analysis (shape matters for recession prediction)
        if 'yield_10y' in economic_data and 'yield_2y' in economic_data:
            yield_sentiment = self._analyze_yield_curve(
                economic_data['yield_10y']['current'],
                economic_data['yield_2y']['current']
            )
            sentiment_factors.append(yield_sentiment)
        
        # Inflation analysis (moderate inflation is good, too high/low is bad)
        if 'inflation' in economic_data:
            # Note: CPIAUCSL is level, need to calculate YoY change for inflation rate
            # For simplicity, using level analysis here
            inflation_sentiment = self._analyze_inflation_level(economic_data['inflation']['current'])
            sentiment_factors.append(inflation_sentiment)
        
        # Asset-specific weighting
        if asset_type == AssetType.CRYPTO:
            # Crypto is more sensitive to monetary policy and risk appetite
            sentiment_factors = [f * 1.2 if 'yield' in str(f) else f for f in sentiment_factors]
        
        if not sentiment_factors:
            return 0.0
        
        return sum(sentiment_factors) / len(sentiment_factors)
    
    def _analyze_unemployment(self, unemployment_rate: float) -> float:
        """Analyze unemployment rate sentiment"""
        if unemployment_rate < 4.0:
            return 0.3
        elif unemployment_rate < 6.0:
            return 0.1
        elif unemployment_rate < 8.0:
            return -0.1
        else:
            return -0.4
    
    def _analyze_consumer_confidence(self, confidence: float) -> float:
        """Analyze consumer confidence sentiment"""
        if confidence > 100:
            return 0.2
        elif confidence > 80:
            return 0.1
        elif confidence > 60:
            return -0.1
        else:
            return -0.3
    
    def _analyze_yield_curve(self, yield_10y: float, yield_2y: float) -> float:
        """Analyze yield curve shape (10Y - 2Y spread)"""
        yield_spread = yield_10y - yield_2y
        
        if yield_spread > 1.0:
            return 0.2
        elif yield_spread > 0:
            return 0.0
        else:
            return -0.4
    
    def _analyze_inflation_level(self, cpi_level: float) -> float:
        """Analyze inflation level (simplified analysis)"""
        if 250 <= cpi_level <= 280:
            return 0.1
        else:
            return -0.1
    
    def _get_relevant_factors(self, asset_type: AssetType) -> List[str]:
        """Get factors most relevant to the asset type"""
        if asset_type == AssetType.CRYPTO:
            return self.crypto_indicators
        else:
            return self.stock_indicators
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        return {
            "indicators_tracked": list(self.indicators.keys()),
            "crypto_factors": self.crypto_indicators,
            "stock_factors": self.stock_indicators,
            "requires_api_key": True,
            "api_key_set": self.is_available(),
            "cost": "Completely free, unlimited requests",
            "signup_url": "https://fred.stlouisfed.org/docs/api/api_key.html",
            "data_source": "Federal Reserve Economic Data"
        }