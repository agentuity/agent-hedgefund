"""
Type conversion utilities for the Hedge Fund Agent
Handles conversion between different type systems used by various modules
"""

from agents.hedge_fund.tools import AssetType
from agents.hedge_fund.agents.trade_decision_agent import AssetType as TradeAssetType


def convert_asset_type(module_asset_type: AssetType) -> TradeAssetType:
    """
    Convert module AssetType to trade decision AssetType
    
    Args:
        module_asset_type: AssetType from the tools module
        
    Returns:
        TradeAssetType for the trade decision agent
    """
    if module_asset_type == AssetType.CRYPTO:
        return TradeAssetType.CRYPTO
    else:
        return TradeAssetType.STOCK 