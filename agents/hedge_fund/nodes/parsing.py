"""
Query Parsing Node

Handles user query parsing using the Enhanced Action Parser.
Extracts intent, assets, portfolio context, and determines next routing action.
"""

import logging
from typing import Dict, Any

from agents.hedge_fund.models import HedgeFundState, NextAction, ParsedAction, IntentType

from agents.hedge_fund.tools.action_parser import parse_user_query

logger = logging.getLogger(__name__)

def parse_query_node(state: HedgeFundState) -> HedgeFundState:
    """Parse user query using the Enhanced Action Parser"""
    try:
        user_query = state["user_query"]
        user_context = state.get("user_context")
        logger.info(f"🔍 Parsing query: {user_query}")
        
        parsed_action = parse_user_query(user_query, user_context)
        state["parsed_action"] = parsed_action
        
        state["next_action"] = determine_routing_decision(parsed_action)
        
        logger.info(f"✅ Query parsed: {parsed_action.intent_type.value}")
        logger.info(f"📋 Next action: {state['next_action']}")
        logger.info(f"🎯 Confidence: {parsed_action.confidence:.1%}")
        logger.info(f"💪 Trade intent strength: {parsed_action.trade_intent_strength:.1%}")
        logger.info(f"📊 Portfolio context: {parsed_action.portfolio_context_strength:.1%}")
        
    except Exception as e:
        state["error"] = f"Query parsing failed: {str(e)}"
        state["next_action"] = NextAction.FORMAT_ERROR
        logger.error(f"❌ Query parsing error: {e}")
    
    return state

def determine_routing_decision(parsed_action: ParsedAction) -> NextAction:
    """
    Determine routing based on extracted information from enhanced parser
    This is where we make DECISIONS based on the EXTRACTED information
    """
    
    if parsed_action.intent_type == IntentType.INVALID_QUERY:
        return NextAction.REJECT_QUERY
    
    if parsed_action.intent_type == IntentType.GENERAL_INFO:
        return NextAction.GENERAL_RESPONSE
    
    if parsed_action.primary_asset or parsed_action.intent_type in [
        IntentType.TRADE_ANALYSIS, 
        IntentType.MARKET_UPDATE,
        IntentType.ASSET_COMPARISON
    ]:
        if not parsed_action.primary_asset:
            return NextAction.CLARIFICATION
        return NextAction.SEARCH_ASSET
    
    if parsed_action.intent_type == IntentType.PORTFOLIO_REVIEW:
        if parsed_action.portfolio_context_strength > 0.7:
            return NextAction.PORTFOLIO_ANALYSIS  # Future: route to portfolio manager
        else:
            return NextAction.CLARIFICATION
    
    if parsed_action.intent_type == IntentType.RISK_ASSESSMENT:
        if parsed_action.risk_concern_level > 0.6:
            return NextAction.RISK_ANALYSIS  # Future: route to risk manager
        else:
            return NextAction.CLARIFICATION
    
    if parsed_action.confidence < 0.5:
        return NextAction.CLARIFICATION
    
    return NextAction.GENERAL_RESPONSE 