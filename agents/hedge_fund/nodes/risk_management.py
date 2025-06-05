"""
Risk Management Node

Handles risk assessment for trade recommendations using user context.
Only triggered when users have actual trade intent, not for general market questions.
"""

import logging

from agents.hedge_fund.models import HedgeFundState, NextAction
from agents.hedge_fund.services.risk_manager_service import assess_trade_risk

logger = logging.getLogger(__name__)

def assess_risk_node(state: HedgeFundState) -> HedgeFundState:
    """Assess risk of trade recommendation against user context"""
    try:
        trade_recommendation = state.get("trade_recommendation")
        parsed_action = state.get("parsed_action")
        
        if not trade_recommendation:
            state["error"] = "No trade recommendation available for risk assessment"
            state["next_action"] = NextAction.FORMAT_ERROR
            return state
        
        if not parsed_action:
            state["error"] = "No parsed action available for risk assessment"
            state["next_action"] = NextAction.FORMAT_ERROR
            return state
        
        logger.info(f"🛡️ Assessing risk for {trade_recommendation.symbol} trade recommendation")
        
        # Perform risk assessment
        risk_assessment = assess_trade_risk(trade_recommendation, parsed_action)
        
        # Store risk assessment results
        state["risk_assessment"] = risk_assessment
        
        # Route to portfolio manager if trade intent exists, otherwise format response
        if _has_trade_intent_after_risk(state):
            logger.info("💼 Trade intent confirmed after risk assessment - routing to portfolio manager")
            state["next_action"] = NextAction.PORTFOLIO_DECISION
        else:
            state["next_action"] = NextAction.FORMAT_RESPONSE
        
        logger.info("✅ Risk assessment complete")
        
        logger.info(f"✅ Risk assessment complete: {risk_assessment.risk_level} risk")
        logger.info(f"🎯 Position size recommendation: {risk_assessment.position_size_recommendation:.1%}")
        logger.info(f"🚦 Should proceed: {risk_assessment.should_proceed}")
        
        # Log warnings for visibility
        if risk_assessment.warnings:
            for warning in risk_assessment.warnings:
                logger.warning(warning)
        
    except Exception as e:
        state["error"] = f"Risk assessment failed: {str(e)}"
        state["next_action"] = NextAction.FORMAT_ERROR
        logger.error(f"❌ Risk assessment error: {e}")
    
    return state 

def _has_trade_intent_after_risk(state: HedgeFundState) -> bool:
    """Check if user still has trade intent after risk assessment"""
    parsed_action = state.get("parsed_action")
    trade_recommendation = state.get("trade_recommendation")
    
    if not parsed_action or not trade_recommendation:
        return False
    
    # Check for strong trade intent
    strong_trade_intent = parsed_action.trade_intent_strength > 0.6
    
    # Check for actionable recommendation
    actionable_recommendation = trade_recommendation.decision.value in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]
    
    return strong_trade_intent and actionable_recommendation 