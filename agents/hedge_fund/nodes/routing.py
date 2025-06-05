"""
Routing Functions

Universal routing logic for the LangGraph workflow.
Maps NextAction enum values to actual node names.
"""

import logging

from agents.hedge_fund.models import NextAction, HedgeFundState

logger = logging.getLogger(__name__)

def route_based_on_next_action(state: HedgeFundState) -> str:
    """
    Universal routing function based on state.next_action
    Handles all routing decisions in one place for cleaner code
    """
    next_action = state.get("next_action", NextAction.FORMAT_ERROR)
    
    # Comprehensive routing map covering all possible next_actions
    routing_map = {
        # From parsing decisions
        NextAction.REJECT_QUERY: "generate_response",
        NextAction.GENERAL_RESPONSE: "generate_response",
        NextAction.CLARIFICATION: "generate_response",
        NextAction.SEARCH_ASSET: "search_asset",
        NextAction.PORTFOLIO_ANALYSIS: "generate_response",  # Future: route to portfolio manager
        NextAction.RISK_ANALYSIS: "generate_response",      # Future: route to risk manager
        
        # From asset search decisions
        NextAction.ANALYZE_TRADE: "analyze_trade",
        NextAction.FORMAT_RESPONSE: "generate_response",
        
        # From analysis decisions  
        NextAction.ASSESS_RISK: "assess_risk",
        NextAction.ANALYZE_PORTFOLIO: "generate_response",  # Future: would route to portfolio manager
        NextAction.PORTFOLIO_DECISION: "portfolio_manager",
        
        # Error handling
        NextAction.FORMAT_ERROR: "format_error"
    }
    
    # Future routing logging
    if next_action == NextAction.ASSESS_RISK:
        logger.info("🛡️ Routing to risk manager")
    elif next_action == NextAction.PORTFOLIO_DECISION:
        logger.info("💼 Routing to portfolio manager")
    elif next_action == NextAction.ANALYZE_PORTFOLIO:
        logger.info("🔮 Future: Would route to portfolio manager")
    
    return routing_map.get(next_action, "format_error") 