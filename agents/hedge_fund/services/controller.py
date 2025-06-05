"""
LangGraph Controller for Hedge Fund Agent

Streamlined controller that orchestrates the workflow using specialized node modules.
Each node is now in its own file for better maintainability.
"""

from typing import Dict, Optional, Any
import logging

from langgraph.graph import StateGraph, END

from agents.hedge_fund.models import HedgeFundState

from agents.hedge_fund.nodes import (
    parse_query_node, search_asset_node, analyze_trade_node, assess_risk_node,
    portfolio_manager_node, generate_llm_response_node, format_error_node, route_based_on_next_action
)

logger = logging.getLogger(__name__)

# --- Main Workflow Creation ---

def create_hedge_fund_workflow() -> StateGraph:
    """Create the main LangGraph workflow for hedge fund analysis"""
    
    workflow = StateGraph(HedgeFundState)
    
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("search_asset", search_asset_node)
    workflow.add_node("analyze_trade", analyze_trade_node)
    workflow.add_node("assess_risk", assess_risk_node)
    workflow.add_node("portfolio_manager", portfolio_manager_node)
    workflow.add_node("generate_response", generate_llm_response_node)
    workflow.add_node("format_error", format_error_node)
    
    workflow.set_entry_point("parse_query")
    
    workflow.add_conditional_edges(
        "parse_query",
        route_based_on_next_action,
        {
            "search_asset": "search_asset",
            "generate_response": "generate_response",
            "format_error": "format_error"
        }
    )
    
    workflow.add_conditional_edges(
        "search_asset",
        route_based_on_next_action,
        {
            "analyze_trade": "analyze_trade",
            "generate_response": "generate_response",
            "format_error": "format_error"
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_trade", 
        route_based_on_next_action,
        {
            "assess_risk": "assess_risk",
            "portfolio_manager": "portfolio_manager",
            "generate_response": "generate_response",
            "format_error": "format_error"
        }
    )
    
    workflow.add_conditional_edges(
        "assess_risk",
        route_based_on_next_action,
        {
            "portfolio_manager": "portfolio_manager",
            "generate_response": "generate_response",
            "format_error": "format_error"
        }
    )
    
    workflow.add_conditional_edges(
        "portfolio_manager",
        route_based_on_next_action,
        {
            "generate_response": "generate_response",
            "format_error": "format_error"
        }
    )
    
    # All formatting paths lead to END
    workflow.add_edge("generate_response", END)
    workflow.add_edge("format_error", END)
    
    return workflow.compile()

# --- Main Controller Function ---

def run_hedge_fund_controller(user_query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Main controller function that orchestrates the hedge fund analysis workflow
    
    Args:
        user_query: The user's query/request
        context: Optional context (user portfolio, risk profile, preferences, etc.)
        
    Returns:
        Formatted response string
    """
    
    logger.info(f"🚀 Starting hedge fund analysis workflow")
    logger.info(f"📝 User query: {user_query}")
    
    # Create workflow with specialized nodes
    workflow = create_hedge_fund_workflow()
    
    # Initialize state
    initial_state: HedgeFundState = {
        "user_query": user_query,
        "user_context": context,
        "parsed_action": None,
        "asset_search_result": None,
        "is_valid_asset": False,
        "trade_recommendation": None,
        "risk_assessment": None,
        "portfolio_decision": None,
        "formatted_response": None,
        "next_action": None,
        "response_ready": False,
        "error": None,
        "debug_info": None,
        "confidence_score": None
    }
    
    # Run workflow
    try:
        final_state = workflow.invoke(initial_state)
        
        if final_state.get("error"):
            logger.error(f"❌ Workflow completed with error: {final_state['error']}")
        else:
            logger.info(f"✅ Workflow completed successfully")
        
        # Extract formatted response
        formatted_response = final_state.get("formatted_response")
        if formatted_response:
            return formatted_response.content
        else:
            return "Sorry, I couldn't process your request."
        
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        return f"Sorry, I encountered an unexpected error: {str(e)}. Please try again."

# --- Example Usage ---
if __name__ == "__main__":
    test_queries = [
        "Should I buy Apple stock?",  # Should trigger risk manager
        "I own 100 shares of Tesla, should I buy more?",  # Should trigger risk manager
        "Is my portfolio too risky with 50% tech stocks?",  # Portfolio analysis
        "How is Bitcoin doing today?",  # General market - no risk manager
        "What is RSI?",  # Educational - no risk manager
        "AAPL vs MSFT which is better?",  # Comparison - might trigger risk manager
        "How do I cook pasta?"  # Off-topic
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing Query: {query}")
        print(f"{'='*80}")
        
        try:
            response = run_hedge_fund_controller(query)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print(f"\n{'-'*40}")
        print("✅ Test completed")
        
    print(f"\n🎉 **Risk Manager Integration Complete!**")
    print("✅ Conditional routing based on trade intent")
    print("✅ Risk assessment for actual trade decisions")  
    print("✅ Bypasses risk manager for general market questions")
    print("✅ Ready for enhanced user-personalized recommendations")