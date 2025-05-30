"""
Hedge Fund Agent
Smart trading assistant that uses LLM to parse queries and real APIs to validate assets
"""

from agentuity import AgentRequest, AgentResponse, AgentContext

# Import tools to use
from agents.hedge_fund.tools import (
    parse_user_query, search_asset, 
    ActionType, TradeIntent, AssetType,
    ParsedAction, AssetInfo
)

# Import trade decision system
from agents.hedge_fund.agents.trade_decision_agent import (
    TradeAnalysisRequest, 
    TradeRecommendation,
    run_trade_decision_analysis,
    AssetType as TradeAssetType
)

def convert_asset_type(module_asset_type: AssetType) -> TradeAssetType:
    """Convert module AssetType to trade decision AssetType"""
    if module_asset_type == AssetType.CRYPTO:
        return TradeAssetType.CRYPTO
    elif module_asset_type == AssetType.STOCK:
        return TradeAssetType.STOCK
    else:
        raise ValueError(f"Unsupported asset type: {module_asset_type}")
def format_trade_recommendation(recommendation: TradeRecommendation, asset_info: AssetInfo,
                                trade_intent: TradeIntent) -> str:
    """Format the trade recommendation into a user-friendly response"""
    
    symbol = recommendation.symbol
    decision = recommendation.decision.value
    confidence = recommendation.confidence.value
    confidence_score = recommendation.confidence_score
    
    # Create intent-specific intro
    if trade_intent == TradeIntent.BUY:
        if decision in ["STRONG_BUY", "BUY"]:
            intro = f"✅ **Good timing for buying {asset_info.name} ({symbol})!**"
        elif decision == "HOLD":
            intro = f"⚠️ **Mixed signals for {asset_info.name} - consider waiting**"
        else:
            intro = f"❌ **Not recommended to buy {asset_info.name} right now**"
    elif trade_intent == TradeIntent.SELL:
        if decision in ["STRONG_SELL", "SELL"]:
            intro = f"✅ **Good timing to sell {asset_info.name}**"
        elif decision == "HOLD":
            intro = f"⚠️ **Mixed signals for {asset_info.name} - consider holding**"
        else:
            intro = f"❌ **Not recommended to sell {asset_info.name} right now**"
    else:
        intro = f"📊 **Analysis for {asset_info.name} ({symbol})**"
    
    # Who doesn't love emojis?
    decision_emoji = {
        "STRONG_BUY": "🚀",
        "BUY": "📈", 
        "HOLD": "⏸️",
        "SELL": "📉",
        "STRONG_SELL": "🔻",
        "NO_TRADE": "⛔"
    }
    
    response_parts = [
        intro,
        "",
        f"{decision_emoji.get(decision, '❓')} **Decision:** {decision}",
        f"🎯 **Confidence:** {confidence} ({confidence_score:.1%})",
        f"💰 **Current Price:** ${asset_info.current_price:.2f}" if asset_info.current_price else "",
        f"🏢 **Exchange:** {asset_info.exchange}" if asset_info.exchange else "",
        "",
        "**Key Factors:**"
    ]
    
    for i, factor in enumerate(recommendation.key_factors[:3], 1):
        response_parts.append(f"{i}. {factor}")
    
    response_parts.extend([
        "",
        f"**Market Context:** {recommendation.market_context}",
        "",
        f"**Analysis:** {recommendation.reasoning}",
        "",
        f"*Analysis completed at {recommendation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*"
    ])
    
    return "\n".join(filter(None, response_parts))

def handle_general_question(action: ParsedAction) -> str:
    """Handle general financial questions"""
    
    general_responses = {
        "market": "The stock market is a platform where shares of publicly traded companies are bought and sold. I can help you analyze specific stocks or crypto assets for trading decisions.",
        "trading": "Trading involves buying and selling financial assets to profit from price movements. I can analyze specific assets and provide trade recommendations based on technical and sentiment analysis.",
        "investing": "Investing is the practice of putting money into assets with the expectation of generating returns over time. I can help you analyze specific stocks or crypto for investment decisions.",
        "crypto": "Cryptocurrency is digital currency secured by cryptography. I can analyze specific crypto assets like Bitcoin, Ethereum, etc. for trading decisions.",
        "stocks": "Stocks represent ownership shares in companies. I can analyze specific stocks and provide buy/sell/hold recommendations based on technical and market sentiment analysis."
    }
    
    query_lower = action.user_message.lower()
    for keyword, response in general_responses.items():
        if keyword in query_lower:
            return f"📚 **{response}**\n\nTry asking about a specific asset like 'Should I buy Elon Musk stock?' or 'How is Bitcoin looking?'"
    
    return "I'm a hedge fund trading assistant. I can help you analyze stocks and crypto assets for trading decisions. Try asking about a specific asset like 'Should I buy Tesla?' or 'How is Ethereum doing?'"

def handle_asset_not_found(action: ParsedAction) -> str:
    """Handle when asset search fails"""
    
    return f"❌ **Asset Not Found**\n\nI couldn't find '{action.asset_query}' in the market data.\n\n**Suggestions:**\n• Try a different spelling or the stock symbol (e.g., 'AAPL' for Apple)\n• Make sure it's a publicly traded company or major cryptocurrency\n• Ask about popular assets like Tesla, Apple, Bitcoin, Ethereum, etc.\n\n*I can analyze major stocks and cryptocurrencies that are actively traded.*"

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """Main agent function - processes user queries and provides trade analysis"""
    
    try:
        user_query = (await request.data.text()).strip()
        
        if not user_query:
            return response.text("Please provide a message asking about a stock or crypto asset. For example: 'Should I buy Tesla?' or 'How is Bitcoin looking?'")
        
        context.logger.info(f"Processing user query: {user_query}")
        
        # Step 1: Parse the user query with LLM
        action = parse_user_query(user_query)
        context.logger.info(f"Parsed action: {action.action_type.value} - {action.reasoning}")
        
        # Step 2: Handle different action types
        if action.action_type == ActionType.REFUSE_QUERY:
            return response.text("I'm a hedge fund trading assistant focused on stock and crypto analysis. I can't help with non-financial topics. Try asking about a specific asset like 'Should I buy Apple stock?' or 'How is Bitcoin doing?'")
        
        elif action.action_type == ActionType.GENERAL_QUESTION:
            general_response = handle_general_question(action)
            return response.text(general_response)
        
        elif action.action_type == ActionType.SEARCH_ASSET:
            if not action.asset_query:
                return response.text("I need to know which asset you're asking about. Please specify a stock or crypto asset.")
            
            # Step 3: Search for the asset using real APIs
            context.logger.info(f"Searching for asset: {action.asset_query}")
            asset_info = search_asset(action.asset_query)
            
            if not asset_info:
                return response.text(handle_asset_not_found(action))
            
            context.logger.info(f"Found asset: {asset_info.name} ({asset_info.symbol}) - {asset_info.asset_type.value}")
            
            # Step 4: Run trade analysis
            try:
                trade_asset_type = convert_asset_type(asset_info.asset_type)
                
                trade_request = TradeAnalysisRequest(
                    symbol=asset_info.symbol,
                    asset_type=trade_asset_type,
                    timeframe="1d"
                )
                
                recommendation = run_trade_decision_analysis(trade_request)
                
                # Step 5: Format and return the response
                formatted_response = format_trade_recommendation(
                    recommendation, 
                    asset_info, 
                    action.trade_intent
                )
                
                return response.text(formatted_response)
                
            except Exception as e:
                context.logger.error(f"Error running trade analysis: {e}")
                return response.text(f"Sorry, I encountered an error analyzing {asset_info.name}. The asset exists but I couldn't complete the analysis. Please try again later.")
        
        elif action.action_type == ActionType.ANALYZE_TRADE:
            # This shouldn't happen in the current flow, but handle it gracefully
            return response.text("I need to search for the asset first. Please specify which stock or crypto you'd like me to analyze.")
        
        else:
            return response.text("I'm not sure how to help with that. Please ask about a specific stock or crypto asset for trading analysis.")
            
    except Exception as e:
        context.logger.error(f"Error in hedge fund agent: {e}")
        return response.text("Sorry, I encountered an unexpected error. Please try again later.")


def controller(request: AgentRequest, response: AgentResponse, context: AgentContext):
    """
    Controller function for the hedge fund agent.
    Routes requests to the main async handler.
    """
    import asyncio
    return asyncio.run(run(request, response, context))