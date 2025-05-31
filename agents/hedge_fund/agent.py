"""
Hedge Fund Agent - Networking Layer

This agent acts as a thin networking hub that receives requests from users
and delegates all processing to the LangGraph controller for orchestration.

Clean, simple, single responsibility: networking only.
"""

from agentuity import AgentRequest, AgentResponse, AgentContext

# Import the new controller
from agents.hedge_fund.agents.controller import run_hedge_fund_controller

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """
    Main agent function - acts as a pure networking layer
    
    Receives user requests and delegates to the LangGraph controller for all processing.
    """
    
    try:
        # Extract user query
        user_query = (await request.data.text()).strip()
        
        if not user_query:
            return response.text(
                "Please provide a message asking about a stock or crypto asset. "
                "For example: 'Should I buy Tesla?' or 'How is Bitcoin looking?'"
            )
        
        context.logger.info(f"📨 Received query: {user_query}")
        
        # Delegate everything to the controller
        context.logger.info("🚀 Delegating to LangGraph controller...")
        formatted_response = run_hedge_fund_controller(user_query)
        
        context.logger.info("✅ Controller completed successfully")
        return response.text(formatted_response)
        
    except Exception as e:
        context.logger.error(f"❌ Error in networking layer: {e}")
        return response.text(
            "Sorry, I encountered an unexpected error. Please try again later."
        )

def controller(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """
    Controller function for the hedge fund agent.
    Routes requests to the main async handler.
    """
    import asyncio
    return asyncio.run(run(request, response, context))