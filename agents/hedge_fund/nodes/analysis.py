"""
Trade Analysis Node

Handles trade analysis using the existing trade decision agent.
Performs technical and sentiment analysis to generate recommendations.
"""

import logging

from agents.hedge_fund.models import NextAction, HedgeFundState

from agents.hedge_fund.services.trade_decision_service import (
    TradeAnalysisRequest, run_trade_decision_analysis,
    AssetType as TradeAssetType
)
from agents.hedge_fund.tools.asset_search import AssetType

logger = logging.getLogger(__name__)

def convert_asset_type(module_asset_type: AssetType) -> TradeAssetType:
    """Convert module AssetType to trade decision AssetType"""
    if module_asset_type == AssetType.CRYPTO:
        return TradeAssetType.CRYPTO
    if module_asset_type in [AssetType.STOCK, AssetType.ETF, AssetType.INDEX]:
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
            market_conditions=None,
            parsed_action=parsed_action
        )
        
        # Run the existing trade decision analysis
        recommendation = run_trade_decision_analysis(trade_request)
        state["trade_recommendation"] = recommendation
        
        logger.info(f"✅ Trade analysis complete: {recommendation.decision.value} with {recommendation.confidence.value} confidence")
        
        # ENHANCED ROUTING: Check if we should route to risk manager
        if _should_route_to_risk_manager(parsed_action, recommendation):
            logger.info("🛡️ Routing to risk manager for trade intent assessment")
            state["next_action"] = NextAction.ASSESS_RISK
        elif _has_trade_intent(parsed_action, recommendation):
            logger.info("💼 Trade intent detected - routing to portfolio manager")
            state["next_action"] = NextAction.PORTFOLIO_DECISION
        else:
            state["next_action"] = NextAction.FORMAT_RESPONSE
        
    except Exception as e:
        state["error"] = f"Trade analysis failed: {str(e)}"
        state["next_action"] = NextAction.FORMAT_ERROR
        logger.error(f"❌ Trade analysis error: {e}")
    
    return state

def _should_route_to_risk_manager(parsed_action, recommendation) -> bool:
    """Determine if we should route to risk manager based on trade intent"""
    
    # Route to risk manager if:
    # 1. Strong trade intent (user wants to actually trade)
    # 2. Actual buy/sell recommendation (not just informational)
    # 3. User expressed risk concerns
    
    strong_trade_intent = parsed_action.trade_intent_strength > 0.6
    actionable_recommendation = recommendation.decision.value in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]
    user_risk_concerns = parsed_action.risk_concern_level > 0.3
    
    # Always route for strong trade intent + actionable recommendation
    if strong_trade_intent and actionable_recommendation:
        return True
    
    # Route if user has risk concerns, even with moderate trade intent
    if user_risk_concerns and actionable_recommendation:
        return True
    
    # Route if user mentioned existing positions (concentration risk check)
    if parsed_action.existing_positions and actionable_recommendation:
        return True
    
    return False

def _has_trade_intent(parsed_action, recommendation) -> bool:
    """Check if user has trade intent and should go to portfolio manager"""
    
    # Strong trade intent from user query
    strong_trade_intent = parsed_action.trade_intent_strength > 0.6
    
    # Actionable recommendation from analysis
    actionable_recommendation = recommendation.decision.value in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]
    
    # Portfolio context suggests they want trading advice
    portfolio_context = parsed_action.portfolio_context_strength > 0.5
    
    return strong_trade_intent and actionable_recommendation and portfolio_context