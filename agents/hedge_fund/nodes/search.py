"""
Asset Search Node

Handles asset search and validation using the Asset Search Agent.
Validates tradeable assets and determines if analysis can proceed.
"""

import logging

# Import from types package to avoid circular imports  
from agents.hedge_fund.models import NextAction, HedgeFundState

# Import specialized agents using full paths from root
from agents.hedge_fund.agents.asset_search_agent import (
    search_and_validate_asset, is_tradeable_asset, get_asset_summary
)

# Set up logger
logger = logging.getLogger(__name__)

def search_asset_node(state: HedgeFundState) -> HedgeFundState:
    """Search for asset using the Asset Search Agent"""
    try:
        parsed_action = state["parsed_action"]
        if not parsed_action or not parsed_action.primary_asset:
            state["error"] = "No asset query provided"
            state["next_action"] = NextAction.FORMAT_ERROR
            return state
        
        asset_query = parsed_action.primary_asset
        logger.info(f"🔎 Searching for asset: {asset_query}")
        
        # Delegate to Asset Search Agent
        search_result = search_and_validate_asset(asset_query, require_validation=True)
        state["asset_search_result"] = search_result
        
        if search_result.found:
            logger.info(f"✅ Asset found: {get_asset_summary(search_result)}")
            
            if is_tradeable_asset(search_result):
                state["next_action"] = NextAction.ANALYZE_TRADE
            else:
                logger.warning(f"⚠️ Asset not suitable for trading analysis")
                state["error"] = "Asset data insufficient for trading analysis"
                state["next_action"] = NextAction.FORMAT_ERROR
        else:
            logger.warning(f"⚠️ Asset not found: {asset_query}")
            state["next_action"] = NextAction.FORMAT_RESPONSE
        
    except Exception as e:
        state["error"] = f"Asset search failed: {str(e)}"
        state["next_action"] = NextAction.FORMAT_ERROR
        logger.error(f"❌ Asset search error: {e}")
    
    return state 