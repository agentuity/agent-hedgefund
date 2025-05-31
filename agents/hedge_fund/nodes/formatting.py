"""
Response Formatting Nodes

Handles response formatting using the Response Formatter Agent.
Contains both success and error response formatting.
"""

import logging

from agents.hedge_fund.models import NextAction, HedgeFundState, ParsedAction, IntentType

from agents.hedge_fund.agents.response_formatter_agent import (
    format_response, get_response_summary, is_actionable_response, FormattedResponse
)

logger = logging.getLogger(__name__)

def format_response_node(state: HedgeFundState) -> HedgeFundState:
    """Format the final response using the Response Formatter Agent"""
    try:
        parsed_action = state["parsed_action"]
        
        if not parsed_action:
            state["error"] = "No parsed action available for formatting"
            state["next_action"] = NextAction.FORMAT_ERROR
            return state
        
        logger.info(f"📝 Formatting response for intent type: {parsed_action.intent_type.value}")
        
        # Delegate to Response Formatter Agent
        formatted_response = format_response(
            parsed_action=parsed_action,
            asset_search_result=state.get("asset_search_result"),
            trade_recommendation=state.get("trade_recommendation"),
            error_message=state.get("error")
        )
        
        state["formatted_response"] = formatted_response
        state["response_ready"] = True
        
        logger.info(f"✅ Response formatted: {get_response_summary(formatted_response)}")
        logger.info(f"📋 Actionable: {is_actionable_response(formatted_response)}")
        
    except Exception as e:
        state["error"] = f"Response formatting failed: {str(e)}"
        state["next_action"] = NextAction.FORMAT_ERROR
        logger.error(f"❌ Response formatting error: {e}")
    
    return state

def format_error_node(state: HedgeFundState) -> HedgeFundState:
    """Format error responses"""
    try:
        error_message = state.get("error", "Unknown error occurred")
        logger.info(f"📝 Formatting error response: {error_message}")
        
        # Create basic parsed action for error formatting if needed
        if not state.get("parsed_action"):
            state["parsed_action"] = ParsedAction(
                original_query=state["user_query"],
                intent_type=IntentType.INVALID_QUERY,
                confidence=0.0,
                reasoning="Error during processing",
                primary_asset=None,
                additional_assets=[],
                mentioned_positions=[],
                quantities=[],
                time_references=[],
                risk_keywords=[],
                market_events=[],
                trade_intent_strength=0.0,
                portfolio_context_strength=0.0,
                risk_concern_level=0.0,
                urgency_level=0.0
            )
        
        # Format error response
        formatted_response = format_response(
            parsed_action=state["parsed_action"],
            error_message=error_message
        )
        
        state["formatted_response"] = formatted_response
        state["response_ready"] = True
        
        logger.info(f"✅ Error response formatted")
        
    except Exception as e:
        # Fallback error handling
        state["formatted_response"] = FormattedResponse(
            content=f"Sorry, I encountered an unexpected error: {str(e)}. Please try again later.",
            response_type="critical_error"
        )
        state["response_ready"] = True
        logger.error(f"❌ Critical error in error formatting: {e}")
    
    return state 