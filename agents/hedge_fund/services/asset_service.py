"""
Asset service for the Hedge Fund Agent
Handles asset search and validation operations
"""

from typing import Optional
from agents.hedge_fund.tools import search_asset, AssetInfo


class AssetService:
    """
    Service responsible for asset search and validation
    Encapsulates asset-related operations
    """
    
    def search_asset(self, asset_query: str) -> Optional[AssetInfo]:
        """
        Search for an asset using the asset query
        
        Args:
            asset_query: Asset name, symbol, or description to search for
            
        Returns:
            AssetInfo if found, None if not found
        """
        if not asset_query or not asset_query.strip():
            return None
        
        return search_asset(asset_query.strip())
    
    def validate_asset_for_trading(self, asset_info: AssetInfo) -> bool:
        """
        Validate if an asset is suitable for trading analysis
        
        Args:
            asset_info: Asset information to validate
            
        Returns:
            True if asset is valid for trading, False otherwise
        """
        if not asset_info:
            return False
        
        # Basic validation - asset should have required fields
        required_fields = [asset_info.symbol, asset_info.name, asset_info.asset_type]
        return all(field is not None for field in required_fields)
    
    def get_asset_display_name(self, asset_info: AssetInfo) -> str:
        """
        Get a user-friendly display name for the asset
        
        Args:
            asset_info: Asset information
            
        Returns:
            Formatted display name
        """
        if not asset_info:
            return "Unknown Asset"
        
        if asset_info.name and asset_info.symbol:
            return f"{asset_info.name} ({asset_info.symbol})"
        elif asset_info.name:
            return asset_info.name
        elif asset_info.symbol:
            return asset_info.symbol
        else:
            return "Unknown Asset" 