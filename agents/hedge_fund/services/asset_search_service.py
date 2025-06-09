"""
Asset Search Agent

Specialized agent for searching and validating financial assets.
Handles both stocks and cryptocurrencies using real market APIs.
"""

from typing import Optional, Dict, Any, List
import logging

from pydantic import BaseModel, Field

from agents.hedge_fund.tools.asset_search import (
    AssetSearcher, AssetInfo, AssetType, search_asset, validate_asset
)

logger = logging.getLogger(__name__)

# --- Search Result Models ---

class AssetSearchResult(BaseModel):
    """Result of asset search operation"""
    query: str
    found: bool
    asset_info: Optional[AssetInfo] = None
    error: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    search_metadata: Dict[str, Any] = Field(default_factory=dict)

class AssetValidationResult(BaseModel):
    """Result of asset validation"""
    asset_info: AssetInfo
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
    market_status: str = "unknown"  # "open", "closed", "pre_market", "after_hours"
    data_availability: Dict[str, bool] = Field(default_factory=dict)

# --- Helper Functions ---

def generate_search_suggestions(failed_query: str) -> List[str]:
    """Generate helpful suggestions when asset search fails"""
    suggestions = []
    
    # Common ticker suggestions
    common_mappings = {
        'apple': 'AAPL',
        'tesla': 'TSLA', 
        'microsoft': 'MSFT',
        'google': 'GOOGL',
        'amazon': 'AMZN',
        'meta': 'META',
        'facebook': 'META',
        'nvidia': 'NVDA',
        'bitcoin': 'BTC-USD',
        'ethereum': 'ETH-USD',
        'dogecoin': 'DOGE-USD'
    }
    
    query_lower = failed_query.lower().strip()
    
    # Check for common name matches
    for name, ticker in common_mappings.items():
        if name in query_lower or query_lower in name:
            suggestions.append(f"Try '{ticker}' for {name.title()}")
    
    # General suggestions
    if not suggestions:
        suggestions.extend([
            f"Try using the stock symbol instead (e.g., 'AAPL' for Apple)",
            f"Make sure it's a publicly traded company or major cryptocurrency",
            f"Check the spelling of the company/asset name"
        ])
    
    # Add popular examples
    suggestions.append("Popular assets: AAPL, TSLA, BTC-USD, ETH-USD, MSFT, GOOGL")
    
    return suggestions[:5]  # Limit to 5 suggestions

def determine_asset_category(asset_info: AssetInfo) -> str:
    """Determine the category of the asset for routing purposes"""
    if asset_info.asset_type == AssetType.CRYPTO:
        return "cryptocurrency"
    elif asset_info.asset_type == AssetType.STOCK:
        # Could further categorize stocks by sector, market cap, etc.
        return "stock"
    else:
        return "unknown"

def check_data_availability(asset_info: AssetInfo) -> Dict[str, bool]:
    """Check what types of data are available for this asset"""
    availability = {
        "price_data": False,
        "volume_data": False,
        "technical_indicators": False,
        "news_sentiment": False,
        "options_data": False
    }
    
    # For now, assume basic data is available if we found the asset
    if asset_info.is_valid:
        availability["price_data"] = True
        availability["volume_data"] = True
        availability["technical_indicators"] = True
        
        # News sentiment usually available for major assets
        if asset_info.asset_type == AssetType.STOCK:
            availability["news_sentiment"] = True
            # Options data mainly for stocks
            availability["options_data"] = True
        elif asset_info.asset_type == AssetType.CRYPTO:
            # Limited news for crypto, no options
            availability["news_sentiment"] = asset_info.symbol in ['BTC-USD', 'ETH-USD']
            availability["options_data"] = False
    
    return availability

# --- Main Search Functions ---

def search_and_validate_asset(asset_query: str, require_validation: bool = True) -> AssetSearchResult:
    """
    Search for an asset and optionally validate it
    
    Args:
        asset_query: The asset query string (symbol, name, etc.)
        require_validation: Whether to perform additional validation
        
    Returns:
        AssetSearchResult with search outcome and details
    """
    
    try:
        logger.info(f"🔎 Searching for asset: {asset_query}")
        
        # Step 1: Basic asset search
        asset_info = search_asset(asset_query)
        
        if not asset_info:
            logger.warning(f"⚠️ Asset not found: {asset_query}")
            suggestions = generate_search_suggestions(asset_query)
            
            return AssetSearchResult(
                query=asset_query,
                found=False,
                error=f"Asset '{asset_query}' not found in market data",
                suggestions=suggestions,
                search_metadata={
                    "search_method": "api_lookup",
                    "attempted_query": asset_query
                }
            )
        
        logger.info(f"✅ Asset found: {asset_info.name} ({asset_info.symbol}) - {asset_info.asset_type.value}")
        
        # Step 2: Optional validation
        validation_result = None
        if require_validation:
            validation_result = validate_asset_data(asset_info)
            if not validation_result.is_valid:
                logger.warning(f"⚠️ Asset validation failed: {validation_result.validation_errors}")
        
        # Step 3: Build successful result
        search_metadata = {
            "search_method": "api_lookup",
            "asset_category": determine_asset_category(asset_info),
            "data_availability": check_data_availability(asset_info),
            "validation_performed": require_validation,
            "validation_passed": validation_result.is_valid if validation_result else None
        }
        
        return AssetSearchResult(
            query=asset_query,
            found=True,
            asset_info=asset_info,
            search_metadata=search_metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Asset search failed: {e}")
        
        return AssetSearchResult(
            query=asset_query,
            found=False,
            error=f"Asset search failed: {str(e)}",
            suggestions=["Please try again later", "Check your internet connection"],
            search_metadata={
                "search_method": "api_lookup",
                "error_type": type(e).__name__
            }
        )

def validate_asset_data(asset_info: AssetInfo) -> AssetValidationResult:
    """
    Validate that asset data is complete and usable for analysis
    
    Args:
        asset_info: The asset information to validate
        
    Returns:
        AssetValidationResult with validation details
    """
    
    try:
        logger.info(f"🔍 Validating asset data for {asset_info.symbol}")
        
        validation_errors = []
        
        # Check required fields
        if not asset_info.symbol:
            validation_errors.append("Missing asset symbol")
        
        if not asset_info.name:
            validation_errors.append("Missing asset name")
        
        if not asset_info.asset_type:
            validation_errors.append("Missing asset type")
        
        # Check asset-specific requirements
        if asset_info.asset_type == AssetType.STOCK:
            if not asset_info.exchange:
                validation_errors.append("Missing exchange information for stock")
        
        # Use existing validation if available
        is_api_valid = validate_asset(asset_info.symbol, asset_info.asset_type)
        if not is_api_valid:
            validation_errors.append("Asset failed API validation")
        
        is_valid = len(validation_errors) == 0
        
        # Determine market status (simplified)
        market_status = "unknown"
        if asset_info.asset_type == AssetType.CRYPTO:
            market_status = "24_7_trading"
        elif asset_info.asset_type == AssetType.STOCK:
            market_status = "market_hours"  # Could be enhanced with real-time market status
        
        data_availability = check_data_availability(asset_info)
        
        result = AssetValidationResult(
            asset_info=asset_info,
            is_valid=is_valid,
            validation_errors=validation_errors,
            market_status=market_status,
            data_availability=data_availability
        )
        
        if is_valid:
            logger.info(f"✅ Asset validation passed for {asset_info.symbol}")
        else:
            logger.warning(f"⚠️ Asset validation failed: {validation_errors}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Asset validation error: {e}")
        
        return AssetValidationResult(
            asset_info=asset_info,
            is_valid=False,
            validation_errors=[f"Validation failed: {str(e)}"],
            market_status="unknown",
            data_availability={}
        )

# --- Batch Operations ---

def search_multiple_assets(asset_queries: List[str]) -> List[AssetSearchResult]:
    """
    Search for multiple assets in batch
    
    Args:
        asset_queries: List of asset query strings
        
    Returns:
        List of AssetSearchResult objects
    """
    
    logger.info(f"🔎 Batch searching {len(asset_queries)} assets")
    
    results = []
    for query in asset_queries:
        result = search_and_validate_asset(query)
        results.append(result)
    
    found_count = sum(1 for r in results if r.found)
    logger.info(f"✅ Batch search complete: {found_count}/{len(asset_queries)} found")
    
    return results

# --- Convenience Functions ---

def get_asset_summary(search_result: AssetSearchResult) -> str:
    """Get a human-readable summary of the search result"""
    if not search_result.found:
        return f"Asset '{search_result.query}' not found"
    
    asset = search_result.asset_info
    return f"{asset.name} ({asset.symbol}) - {asset.asset_type.value.title()}"

def is_tradeable_asset(search_result: AssetSearchResult) -> bool:
    """Check if the found asset is suitable for trading analysis"""
    if not search_result.found or not search_result.asset_info:
        return False
    
    # Check if basic data is available
    data_availability = search_result.search_metadata.get("data_availability", {})
    return data_availability.get("price_data", False) and data_availability.get("technical_indicators", False)

def format_asset_not_found_message(search_result: AssetSearchResult) -> str:
    """Format a user-friendly message when asset is not found"""
    if search_result.found:
        return "Asset was found successfully"
    
    message_parts = [
        f"❌ **Asset Not Found**",
        f"",
        f"I couldn't find '{search_result.query}' in the market data.",
        f"",
        f"**Suggestions:**"
    ]
    
    for suggestion in search_result.suggestions:
        message_parts.append(f"• {suggestion}")
    
    message_parts.extend([
        f"",
        f"*I can analyze major stocks and cryptocurrencies that are actively traded.*"
    ])
    
    return "\n".join(message_parts)
