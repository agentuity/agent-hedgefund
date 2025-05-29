"""
Asset Search Module
Uses LLM to convert company names to symbols, then validates with yfinance
"""

import yfinance as yf
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import time
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class AssetType(Enum):
    STOCK = "STOCK"
    CRYPTO = "CRYPTO"
    ETF = "ETF"
    INDEX = "INDEX"
    UNKNOWN = "UNKNOWN"

@dataclass
class AssetInfo:
    """Information about a validated asset"""
    symbol: str
    name: str
    asset_type: AssetType
    exchange: Optional[str] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None
    current_price: Optional[float] = None
    is_valid: bool = True

class SymbolResolution(BaseModel):
    """Structured response for symbol resolution"""
    symbol: str = Field(
        description="The trading symbol for the asset, or 'unknown' if not recognized",
        examples=["AAPL", "BTC-USD", "SPY", "unknown"]
    )
    asset_type: str = Field(
        description="The type of asset",
        examples=["STOCK", "CRYPTO", "ETF", "INDEX", "unknown"]
    )
    confidence: float = Field(
        description="Confidence level in the symbol resolution",
        ge=0.0,
        le=1.0,
        examples=[0.95, 0.87, 0.76]
    )
    reasoning: str = Field(
        description="Explanation of the symbol resolution decision",
        examples=[
            "Apple Inc. trades under symbol AAPL",
            "Bitcoin trades as BTC-USD on Yahoo Finance",
            "Not a recognized publicly traded company"
        ]
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "symbol": "AAPL",
                    "asset_type": "STOCK",
                    "confidence": 0.95,
                    "reasoning": "Apple Inc. trades under symbol AAPL"
                },
                {
                    "symbol": "BTC-USD",
                    "asset_type": "CRYPTO",
                    "confidence": 0.95,
                    "reasoning": "Bitcoin trades as BTC-USD on Yahoo Finance"
                },
                {
                    "symbol": "unknown",
                    "asset_type": "unknown",
                    "confidence": 0.9,
                    "reasoning": "Not a recognized publicly traded company"
                }
            ]
        }

class LLMAssetResolver:
    """Uses LLM to resolve company names to stock/crypto symbols"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        # Use LangChain's structured output functionality
        self.structured_llm = self.llm.with_structured_output(
            SymbolResolution,
            include_raw=True  # Include raw response for debugging if needed
        )
    
    def resolve_asset_symbol(self, query: str) -> Optional[Dict[str, str]]:
        """
        Use LLM to convert company name/query to proper symbol
        
        Returns:
            Dict with 'symbol', 'asset_type', 'reasoning' or None if unknown
        """
        
        try:
            system_prompt = self._get_symbol_resolution_prompt()
            user_prompt = f"What is the trading symbol for: '{query}'"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Use structured output - this handles retries and validation automatically
            result = self.structured_llm.invoke(messages)
            
            # If include_raw=True, result is a dict with 'parsed' and 'raw' keys
            if isinstance(result, dict) and 'parsed' in result:
                resolution = result['parsed']
            else:
                resolution = result
            
            # Convert to dict format for backward compatibility
            return {
                "symbol": resolution.symbol,
                "asset_type": resolution.asset_type,
                "confidence": resolution.confidence,
                "reasoning": resolution.reasoning
            }
                
        except Exception as e:
            print(f"Error in LLM symbol resolution: {e}")
            return None
    
    def _get_symbol_resolution_prompt(self) -> str:
        """Get the system prompt for symbol resolution"""
        
        return """You are a financial symbol resolver. Your job is to convert company names, crypto names, or partial queries into proper trading symbols.

SUPPORTED ASSETS:
Our system supports assets that are available on Yahoo Finance, including:

MAJOR STOCKS:
- Apple (AAPL), Microsoft (MSFT), Google/Alphabet (GOOGL), Amazon (AMZN)
- Tesla (TSLA), Meta/Facebook (META), Netflix (NFLX), Nvidia (NVDA)
- AMD (AMD), Intel (INTC), Berkshire Hathaway (BRK-B)
- Johnson & Johnson (JNJ), Walmart (WMT), Visa (V), Mastercard (MA)
- Coca-Cola (KO), Disney (DIS), McDonald's (MCD), Nike (NKE)
- And thousands of other publicly traded stocks on major exchanges

MAJOR CRYPTOCURRENCIES:
- Bitcoin (BTC-USD), Ethereum (ETH-USD), Solana (SOL-USD)
- Cardano (ADA-USD), Polygon (MATIC-USD), Chainlink (LINK-USD)
- Dogecoin (DOGE-USD), Litecoin (LTC-USD), Avalanche (AVAX-USD)
- And other major cryptocurrencies (use -USD suffix)

MAJOR ETFS/INDICES:
- S&P 500 (SPY), Nasdaq (QQQ), Total Stock Market (VTI)
- Dow Jones (DIA), Russell 2000 (IWM)

RULES:
1. If you recognize the company/asset, provide the correct symbol
2. For crypto, always use the -USD suffix (e.g., BTC-USD, ETH-USD)
3. If you're not confident or don't recognize it, return "unknown"
4. Only suggest symbols for well-known, publicly traded assets

EXAMPLES:

User: "Apple"
→ symbol: "AAPL", asset_type: "STOCK", confidence: 0.95, reasoning: "Apple Inc. trades under symbol AAPL"

User: "Bitcoin"
→ symbol: "BTC-USD", asset_type: "CRYPTO", confidence: 0.95, reasoning: "Bitcoin trades as BTC-USD on Yahoo Finance"

User: "RANDOMCOMPANY123"
→ symbol: "unknown", asset_type: "unknown", confidence: 0.9, reasoning: "Not a recognized publicly traded company"

User: "S&P 500"
→ symbol: "SPY", asset_type: "ETF", confidence: 0.9, reasoning: "SPY is the most popular S&P 500 ETF"

Always provide clear reasoning for your decision."""

class AssetSearcher:
    """Searches and validates assets using LLM + yfinance validation"""
    
    def __init__(self):
        self.cache = {}  # Simple cache to avoid repeated API calls
        self.cache_ttl = 300  # 5 minutes cache
        self.llm_resolver = LLMAssetResolver()
    
    def search_asset(self, query: str) -> Optional[AssetInfo]:
        """
        Search for an asset using LLM resolution + yfinance validation
        
        Args:
            query: Company name, symbol, or crypto name
            
        Returns:
            AssetInfo if found and valid, None if not found
        """
        query = query.strip()
        
        # Check cache first
        cache_key = f"search_{query.upper()}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result
        
        # Step 1: Use LLM to resolve the query to a symbol
        resolution = self.llm_resolver.resolve_asset_symbol(query)
        
        if not resolution or resolution.get("symbol") == "unknown":
            # Cache negative result
            self.cache[cache_key] = (time.time(), None)
            return None
        
        symbol = resolution["symbol"]
        suggested_type = resolution["asset_type"]
        
        # Step 2: Validate the symbol with yfinance
        asset_info = self._validate_symbol_with_yfinance(symbol, suggested_type)
        
        # Cache the result
        self.cache[cache_key] = (time.time(), asset_info)
        return asset_info
    
    def _validate_symbol_with_yfinance(self, symbol: str, suggested_type: str) -> Optional[AssetInfo]:
        """Validate a symbol using yfinance and get detailed info"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if it's a valid asset
            if not info or len(info) < 5:  # yfinance returns minimal dict for invalid symbols
                return None
            
            # Get basic info
            name = info.get('longName', info.get('shortName', symbol))
            if not name or name == symbol:
                # If we can't get a proper name, it might be invalid
                return None
            
            exchange = info.get('exchange', 'Unknown')
            currency = info.get('currency', 'USD')
            market_cap = info.get('marketCap')
            current_price = info.get('currentPrice', info.get('regularMarketPrice'))
            
            # Determine actual asset type based on yfinance data
            asset_type = self._determine_asset_type(symbol, info, suggested_type)
            
            return AssetInfo(
                symbol=symbol,
                name=name,
                asset_type=asset_type,
                exchange=exchange,
                currency=currency,
                market_cap=market_cap,
                current_price=current_price,
                is_valid=True
            )
            
        except Exception as e:
            print(f"Error validating symbol {symbol}: {e}")
            return None
    
    def _determine_asset_type(self, symbol: str, info: dict, suggested_type: str) -> AssetType:
        """Determine the actual asset type from yfinance data"""
        
        # Check for crypto (usually has -USD suffix and specific quote type)
        if symbol.endswith('-USD') or info.get('quoteType') == 'CRYPTOCURRENCY':
            return AssetType.CRYPTO
        
        # Check for ETF
        quote_type = info.get('quoteType', '').upper()
        if quote_type == 'ETF' or 'ETF' in info.get('longName', '').upper():
            return AssetType.ETF
        
        # Check for common index ETFs
        if symbol in ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VEA', 'VWO']:
            return AssetType.INDEX
        
        # Default to stock
        return AssetType.STOCK

# Global instance
asset_searcher = AssetSearcher()

def search_asset(query: str) -> Optional[AssetInfo]:
    """Convenience function to search for an asset"""
    return asset_searcher.search_asset(query)

def validate_asset(symbol: str, asset_type: AssetType) -> Optional[AssetInfo]:
    """Validate a specific asset symbol"""
    return asset_searcher._validate_symbol_with_yfinance(symbol, asset_type.value) 