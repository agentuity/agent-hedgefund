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
        state["risk_assessment"] = risk_assessment
        
        logger.info(f"✅ Risk assessment complete: {risk_assessment.risk_level} risk")
        logger.info(f"🎯 Position size recommendation: {risk_assessment.position_size_recommendation:.1%}")
        logger.info(f"🚦 Should proceed: {risk_assessment.should_proceed}")
        
        # Log warnings for visibility
        if risk_assessment.warnings:
            for warning in risk_assessment.warnings:
                logger.warning(warning)
        
        # Always proceed to formatting (risk info will be included in response)
        state["next_action"] = NextAction.FORMAT_RESPONSE
        
    except Exception as e:
        state["error"] = f"Risk assessment failed: {str(e)}"
        state["next_action"] = NextAction.FORMAT_ERROR
        logger.error(f"❌ Risk assessment error: {e}")
    
    return state 