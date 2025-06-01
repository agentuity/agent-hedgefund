"""
Trade Analysis Node

Handles trade analysis using the existing trade decision agent.
Performs technical and sentiment analysis to generate recommendations.
"""

import logging

from agents.hedge_fund.models import NextAction, HedgeFundState

from agents.hedge_fund.services.trade_decision_agent import (
    TradeAnalysisRequest, run_trade_decision_analysis,
    AssetType as TradeAssetType
)
from agents.hedge_fund.tools.asset_search import AssetType

logger = logging.getLogger(__name__)

def convert_asset_type(module_asset_type: AssetType) -> TradeAssetType:
    """Convert module AssetType to trade decision AssetType"""
    if module_asset_type == AssetType.CRYPTO:
        return TradeAssetType.CRYPTO
    elif module_asset_type == AssetType.STOCK:
        return TradeAssetType.STOCK
    else:
        raise ValueError(f"Unsupported asset type: {module_asset_type}")

def analyze_trade_node(state: HedgeFundState) -> HedgeFundState:
    """Run trade analysis using the existing trade decision agent"""
    try:
        search_result = state["asset_search_result"]
        parsed_action = state["parsed_action"]
        
        if not search_result or not search_result.found or not search_result.asset_info:
            state["error"] = "No valid asset information available for trade analysis"
            state["next_action"] = NextAction.FORMAT_ERROR
            return state
        
        asset_info = search_result.asset_info
        logger.info(f"📊 Running trade analysis for {asset_info.symbol}")
        
        # Convert asset type and create trade request
        trade_asset_type = convert_asset_type(asset_info.asset_type)
        
        trade_request = TradeAnalysisRequest(
            symbol=asset_info.symbol,
            asset_type=trade_asset_type,
            timeframe="1d",
            use_enhanced_sentiment=True,  # Enable LLM-powered sentiment analysis
            current_portfolio_context=state.get("user_context"),
            market_conditions=None  # Could be enhanced with market data
        )
        
        # Run the existing trade decision analysis
        recommendation = run_trade_decision_analysis(trade_request)
        state["trade_recommendation"] = recommendation
        
        logger.info(f"✅ Trade analysis complete: {recommendation.decision.value} with {recommendation.confidence.value} confidence")
        
        # Check if we need risk management based on extracted information
        if parsed_action.risk_concern_level > 0.6:
            state["next_action"] = NextAction.ASSESS_RISK  # Future: route to risk manager
        elif parsed_action.portfolio_context_strength > 0.7:
            state["next_action"] = NextAction.ANALYZE_PORTFOLIO  # Future: route to portfolio manager
        else:
            state["next_action"] = NextAction.FORMAT_RESPONSE
        
    except Exception as e:
        state["error"] = f"Trade analysis failed: {str(e)}"
        state["next_action"] = NextAction.FORMAT_ERROR
        logger.error(f"❌ Trade analysis error: {e}")
    
    return state 