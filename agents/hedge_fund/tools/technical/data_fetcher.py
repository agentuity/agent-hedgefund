"""
Market Data Fetcher Module

Handles fetching market data from various sources (yfinance, CoinGecko).
Clean interface for data retrieval without analysis logic.
Supports stocks, ETFs, index funds, and cryptocurrencies.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import logging

from agents.hedge_fund.tools.sentiment.base import AssetType

# Set up logger
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Clean container for market data"""
    symbol: str
    asset_type: AssetType
    prices: List[float]
    volumes: List[float]
    current_price: Optional[float] = None
    timeframe: str = "1d"
    
    def __post_init__(self):
        """Set current price if not provided"""
        if self.current_price is None and self.prices:
            self.current_price = self.prices[-1]

def fetch_yfinance_data(symbol: str, asset_type: AssetType, timeframe: str = "1d", period: str = "1y") -> Optional[MarketData]:
    """
    Fetch market data using yfinance (supports stocks, ETFs, index funds)
    Returns MarketData object or None if data cannot be fetched
    """
    try:
        asset_name = {
            AssetType.STOCK: "stock",
            AssetType.ETF: "ETF", 
            AssetType.INDEX: "index fund"
        }.get(asset_type, "asset")
        
        logger.info(f"🔍 Fetching {asset_name} data for {symbol}")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=timeframe)
        
        if data.empty:
            logger.warning(f"⚠️ No data returned for {symbol}")
            return None
            
        prices = data['Close'].tolist()
        volumes = data['Volume'].tolist()
        
        if not prices or not volumes:
            logger.warning(f"⚠️ Empty price or volume data for {symbol}")
            return None
            
        logger.info(f"✅ Successfully fetched {len(prices)} data points for {symbol}")
        
        return MarketData(
            symbol=symbol,
            asset_type=asset_type,
            prices=prices,
            volumes=volumes,
            timeframe=timeframe
        )
        
    except Exception as e:
        logger.error(f"❌ Error fetching yfinance data for {symbol}: {e}")
        return None

def fetch_crypto_data(symbol: str, timeframe: str = "1d", days: int = 365) -> Optional[MarketData]:
    """
    Fetch crypto data using CoinGecko API
    Returns MarketData object or None if data cannot be fetched
    """
    try:
        logger.info(f"🔍 Fetching crypto data for {symbol}")
        
        cg = CoinGeckoAPI()
        
        # Map common symbols to CoinGecko IDs
        symbol_mapping = {
            "BTC-USD": "bitcoin",
            "BTC": "bitcoin", 
            "ETH-USD": "ethereum",
            "ETH": "ethereum",
            "ADA-USD": "cardano",
            "ADA": "cardano",
            "SOL-USD": "solana",
            "SOL": "solana",
            "DOGE-USD": "dogecoin",
            "DOGE": "dogecoin",
            "MATIC-USD": "matic-network",
            "MATIC": "matic-network",
            "DOT-USD": "polkadot",
            "DOT": "polkadot",
            "LINK-USD": "chainlink",
            "LINK": "chainlink",
            "AVAX-USD": "avalanche-2",
            "AVAX": "avalanche-2",
            "LTC-USD": "litecoin",
            "LTC": "litecoin"
        }
        
        coin_id = symbol_mapping.get(symbol.upper())
        if not coin_id:
            logger.warning(f"⚠️ Unknown crypto symbol: {symbol}")
            return None
        
        # Fetch market chart data
        market_chart = cg.get_coin_market_chart_by_id(
            id=coin_id, 
            vs_currency='usd', 
            days=days
        )
        
        if not market_chart or 'prices' not in market_chart or 'total_volumes' not in market_chart:
            logger.warning(f"⚠️ No price or volume data returned for {symbol}")
            return None
            
        # Extract prices and volumes
        prices = [price_point[1] for price_point in market_chart['prices']]
        volumes = [volume_point[1] for volume_point in market_chart['total_volumes']]
        
        if not prices or not volumes:
            logger.warning(f"⚠️ Empty price or volume data for {symbol}")
            return None
            
        logger.info(f"✅ Successfully fetched {len(prices)} data points for {symbol}")
        
        return MarketData(
            symbol=symbol,
            asset_type=AssetType.CRYPTO,
            prices=prices,
            volumes=volumes,
            timeframe=timeframe
        )
        
    except Exception as e:
        logger.error(f"❌ Error fetching crypto data for {symbol}: {e}")
        return None

def fetch_market_data(symbol: str, asset_type: AssetType, timeframe: str = "1d") -> Optional[MarketData]:
    """
    Universal market data fetcher
    Automatically routes to appropriate data source based on asset type
    """
    try:
        logger.info(f"🔍 Fetching market data for {symbol} ({asset_type.value})")
        
        if asset_type in [AssetType.STOCK, AssetType.ETF, AssetType.INDEX]:
            # All these asset types are available through yfinance
            return fetch_yfinance_data(symbol, asset_type, timeframe)
        elif asset_type == AssetType.CRYPTO:
            return fetch_crypto_data(symbol, timeframe)
        else:
            logger.error(f"❌ Unsupported asset type: {asset_type}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error in universal data fetcher for {symbol}: {e}")
        return None

# --- Utility Functions ---

def validate_market_data(market_data: MarketData, min_periods: int = 50) -> bool:
    """
    Validate that market data has sufficient data points for analysis
    """
    if not market_data or not market_data.prices or not market_data.volumes:
        return False
    
    if len(market_data.prices) < min_periods:
        logger.warning(f"⚠️ Insufficient data: {len(market_data.prices)} periods (need {min_periods})")
        return False
    
    return True

def get_supported_assets() -> Dict[str, List[str]]:
    """Get lists of supported assets by type"""
    return {
        "popular_etfs": [
            "SPY",   # S&P 500
            "QQQ",   # Nasdaq 100
            "VTI",   # Total Stock Market
            "DIA",   # Dow Jones
            "IWM",   # Russell 2000
            "XLF",   # Financial Sector
            "XLK",   # Technology Sector
            "XLE",   # Energy Sector
            "VEA",   # Developed Markets
            "VWO"    # Emerging Markets
        ],
        "popular_index_funds": [
            "VTIAX", # Vanguard Total International Stock
            "FXAIX", # Fidelity 500 Index Fund
            "SWTSX", # Schwab Total Stock Market
            "VTSAX", # Vanguard Total Stock Market
            "VFIAX"  # Vanguard 500 Index Fund
        ],
        "crypto": [
            "BTC-USD", "BTC", "ETH-USD", "ETH", "ADA-USD", "ADA",
            "SOL-USD", "SOL", "DOGE-USD", "DOGE", "MATIC-USD", "MATIC",
            "DOT-USD", "DOT", "LINK-USD", "LINK", "AVAX-USD", "AVAX",
            "LTC-USD", "LTC"
        ]
    }

def get_data_summary(market_data: MarketData) -> str:
    """Get a brief summary of the market data"""
    if not market_data:
        return "No market data available"
    
    return f"{market_data.symbol} ({market_data.asset_type.value}): {len(market_data.prices)} periods, Current: ${market_data.current_price:.2f}"

def is_etf_or_index(symbol: str) -> bool:
    """Check if a symbol is likely an ETF or index fund"""
    etfs_and_indexes = get_supported_assets()["popular_etfs"] + get_supported_assets()["popular_index_funds"]
    return symbol.upper() in [s.upper() for s in etfs_and_indexes]

# --- Backward Compatibility ---
# Keep the old function name for backward compatibility
def fetch_stock_data(symbol: str, timeframe: str = "1d", period: str = "1y") -> Optional[MarketData]:
    """
    Legacy function for backward compatibility
    Routes to fetch_yfinance_data with STOCK asset type
    """
    return fetch_yfinance_data(symbol, AssetType.STOCK, timeframe, period) 