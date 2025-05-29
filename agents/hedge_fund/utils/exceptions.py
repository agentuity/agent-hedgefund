"""
Custom exceptions for the Hedge Fund Agent
Provides specific exception types for better error handling and debugging
"""


class HedgeFundException(Exception):
    """Base exception for hedge fund agent operations"""
    pass


class QueryParsingException(HedgeFundException):
    """Raised when user query parsing fails"""
    pass


class AssetNotFoundException(HedgeFundException):
    """Raised when an asset cannot be found in market data"""
    def __init__(self, asset_query: str, message: str = None):
        self.asset_query = asset_query
        super().__init__(message or f"Asset '{asset_query}' not found in market data")


class AssetValidationException(HedgeFundException):
    """Raised when asset validation fails"""
    pass


class TradeAnalysisException(HedgeFundException):
    """Raised when trade analysis fails"""
    def __init__(self, symbol: str, message: str = None):
        self.symbol = symbol
        super().__init__(message or f"Trade analysis failed for {symbol}")


class InvalidTradeRequestException(HedgeFundException):
    """Raised when trade request is invalid"""
    pass


class DataFetchException(HedgeFundException):
    """Raised when market data fetching fails"""
    pass 