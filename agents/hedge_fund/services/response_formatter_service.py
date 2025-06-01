"""
Response Formatter Agent

Specialized agent for formatting responses in a user-friendly way.
Handles trade recommendations, error messages, general info, and more.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from pydantic import BaseModel, Field

from agents.hedge_fund.services.trade_decision_agent import TradeRecommendation
from agents.hedge_fund.tools.asset_search import AssetInfo
from agents.hedge_fund.services.asset_search_agent import AssetSearchResult

from agents.hedge_fund.models import ParsedAction, IntentType, TradeDirection

logger = logging.getLogger(__name__)

# --- Response Models ---

class FormattedResponse(BaseModel):
    """Formatted response ready for user"""
    content: str
    response_type: str  # "trade_recommendation", "error", "general_info", etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

# --- Trade Recommendation Formatting ---

def format_trade_recommendation_response(
    recommendation: TradeRecommendation, 
    asset_info: AssetInfo, 
    parsed_action: ParsedAction
) -> FormattedResponse:
    """Format a complete trade recommendation response"""
    
    try:
        logger.info(f"📝 Formatting trade recommendation for {asset_info.symbol}")
        
        symbol = recommendation.symbol
        decision = recommendation.decision.value
        confidence = recommendation.confidence.value
        confidence_score = recommendation.confidence_score
        trade_direction = parsed_action.trade_direction
        
        # Create intent-specific intro
        intro = _create_trade_intro(decision, asset_info, trade_direction)
        
        # Decision emojis for visual appeal
        decision_emoji = {
            "STRONG_BUY": "🚀", "BUY": "📈", "HOLD": "⏸️",
            "SELL": "📉", "STRONG_SELL": "🔻", "NO_TRADE": "⛔"
        }
        
        # Build response parts
        response_parts = [
            intro,
            "",
            f"{decision_emoji.get(decision, '❓')} **Decision:** {decision}",
            f"🎯 **Confidence:** {confidence} ({confidence_score:.1%})",
        ]
        
        # Add price and exchange info if available
        if asset_info.current_price:
            response_parts.append(f"💰 **Current Price:** ${asset_info.current_price:.2f}")
        
        if asset_info.exchange:
            response_parts.append(f"🏢 **Exchange:** {asset_info.exchange}")
        
        # Add key factors
        response_parts.extend(["", "**Key Factors:**"])
        for i, factor in enumerate(recommendation.key_factors[:3], 1):
            response_parts.append(f"{i}. {factor}")
        
        # Add enhanced insights if available
        if recommendation.enhanced_insights:
            response_parts.extend(["", "**AI Insights:**"])
            insights = recommendation.enhanced_insights
            
            if insights.get('key_themes'):
                themes = ', '.join(insights['key_themes'][:3])
                response_parts.append(f"🧠 **Market Themes:** {themes}")
            
            if insights.get('market_moving_events'):
                events = ', '.join(insights['market_moving_events'][:2])
                response_parts.append(f"📰 **Key Events:** {events}")
            
            if insights.get('risk_factors'):
                risks = ', '.join(insights['risk_factors'][:2])
                response_parts.append(f"⚠️ **Risk Factors:** {risks}")
        
        # Add market context and analysis
        response_parts.extend([
            "",
            f"**Market Context:** {recommendation.market_context}",
            "",
            f"**Analysis:** {recommendation.reasoning}"
        ])
        
        # Add portfolio context if detected
        if parsed_action.portfolio_context_strength > 0.5:
            if parsed_action.mentioned_positions:
                response_parts.extend([
                    "",
                    "**Portfolio Context:** Based on your existing positions"
                ])
        
        # Add timestamp
        response_parts.extend([
            "",
            f"*Analysis completed at {recommendation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        content = "\n".join(filter(None, response_parts))
        
        return FormattedResponse(
            content=content,
            response_type="trade_recommendation",
            metadata={
                "symbol": symbol,
                "decision": decision,
                "confidence_score": confidence_score,
                "asset_type": asset_info.asset_type.value,
                "intent_type": parsed_action.intent_type.value
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error formatting trade recommendation: {e}")
        return _format_error_response(f"Failed to format trade recommendation: {str(e)}")

def _create_trade_intro(decision: str, asset_info: AssetInfo, trade_direction: Optional[TradeDirection]) -> str:
    """Create the introductory line based on decision and trade direction"""
    
    name = asset_info.name
    symbol = asset_info.symbol
    
    if trade_direction == TradeDirection.BUY:
        if decision in ["STRONG_BUY", "BUY"]:
            return f"✅ **Good timing for buying {name} ({symbol})!**"
        elif decision == "HOLD":
            return f"⚠️ **Mixed signals for {name} - consider waiting**"
        else:
            return f"❌ **Not recommended to buy {name} right now**"
            
    elif trade_direction == TradeDirection.SELL:
        if decision in ["STRONG_SELL", "SELL"]:
            return f"✅ **Good timing to sell {name}**"
        elif decision == "HOLD":
            return f"⚠️ **Mixed signals for {name} - consider holding**"
        else:
            return f"❌ **Not recommended to sell {name} right now**"
            
    else:
        return f"📊 **Analysis for {name} ({symbol})**"

# --- Error Response Formatting ---

def format_asset_not_found_response(search_result: AssetSearchResult) -> FormattedResponse:
    """Format response when asset is not found"""
    
    logger.info(f"📝 Formatting asset not found response for: {search_result.query}")
    
    message_parts = [
        "❌ **Asset Not Found**",
        "",
        f"I couldn't find '{search_result.query}' in the market data.",
        "",
        "**Suggestions:**"
    ]
    
    for suggestion in search_result.suggestions[:4]:
        message_parts.append(f"• {suggestion}")
    
    message_parts.extend([
        "",
        "*I can analyze major stocks and cryptocurrencies that are actively traded.*"
    ])
    
    content = "\n".join(message_parts)
    
    return FormattedResponse(
        content=content,
        response_type="asset_not_found",
        metadata={
            "failed_query": search_result.query,
            "suggestions_provided": len(search_result.suggestions)
        }
    )

def format_invalid_query_response() -> FormattedResponse:
    """Format response for invalid/rejected queries"""
    
    content = (
        "I'm a hedge fund trading assistant focused on stock and crypto analysis. "
        "I can't help with non-financial topics. Try asking about a specific asset "
        "like 'Should I buy Apple stock?' or 'How is Bitcoin doing?'"
    )
    
    return FormattedResponse(
        content=content,
        response_type="invalid_query",
        metadata={"reason": "non_financial_query"}
    )

def format_general_info_response(parsed_action: ParsedAction) -> FormattedResponse:
    """Format response for general financial questions"""
    
    logger.info("📝 Formatting general info response")
    
    # Try to provide specific responses based on query content
    query_lower = parsed_action.original_query.lower()
    
    general_responses = {
        "market": "The stock market is a platform where shares of publicly traded companies are bought and sold. I can help you analyze specific stocks or crypto assets for trading decisions.",
        "trading": "Trading involves buying and selling financial assets to profit from price movements. I can analyze specific assets and provide trade recommendations based on technical and sentiment analysis.",
        "investing": "Investing is the practice of putting money into assets with the expectation of generating returns over time. I can help you analyze specific stocks or crypto for investment decisions.",
        "crypto": "Cryptocurrency is digital currency secured by cryptography. I can analyze specific crypto assets like Bitcoin, Ethereum, etc. for trading decisions.",
        "stocks": "Stocks represent ownership shares in companies. I can analyze specific stocks and provide buy/sell/hold recommendations based on technical and market sentiment analysis.",
        "rsi": "RSI (Relative Strength Index) is a momentum oscillator that measures the speed and magnitude of price changes. It ranges from 0-100, with values above 70 indicating overbought conditions and below 30 indicating oversold conditions.",
        "macd": "MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.",
        "ema": "EMA (Exponential Moving Average) is a type of moving average that gives more weight to recent prices, making it more responsive to new information than a simple moving average."
    }
    
    # Find matching response
    specific_response = None
    for keyword, response in general_responses.items():
        if keyword in query_lower:
            specific_response = response
            break
    
    if specific_response:
        content = f"📚 **{specific_response}**\n\nTry asking about a specific asset like 'Should I buy Tesla?' or 'How is Bitcoin looking?'"
    else:
        content = "📚 **I'm a hedge fund trading assistant.** I can help you analyze specific stocks and crypto assets for trading decisions.\n\nTry asking about a specific asset like 'Should I buy Tesla?' or 'How is Bitcoin looking?'"
    
    return FormattedResponse(
        content=content,
        response_type="general_info",
        metadata={
            "query_type": "general_financial_info",
            "specific_match": specific_response is not None
        }
    )

def format_clarification_response(parsed_action: ParsedAction) -> FormattedResponse:
    """Format response when query needs clarification"""
    
    content = (
        "I need more information to help you effectively. Could you please:\n\n"
        "• Specify which stock or cryptocurrency you're interested in\n"
        "• Clarify if you want to buy, sell, or just analyze an asset\n"
        "• Be more specific about your trading question\n\n"
        "For example: 'Should I buy Apple stock?' or 'How is Bitcoin performing?'"
    )
    
    return FormattedResponse(
        content=content,
        response_type="clarification_needed",
        metadata={
            "original_query": parsed_action.original_query,
            "parsing_confidence": parsed_action.confidence
        }
    )

def _format_error_response(error_message: str) -> FormattedResponse:
    """Format a generic error response"""
    
    content = f"Sorry, I encountered an error: {error_message}. Please try again later."
    
    return FormattedResponse(
        content=content,
        response_type="error",
        metadata={"error_message": error_message}
    )

# --- Main Formatting Function ---

def format_response(
    parsed_action: ParsedAction,
    asset_search_result: Optional[AssetSearchResult] = None,
    trade_recommendation: Optional[TradeRecommendation] = None,
    error_message: Optional[str] = None
) -> FormattedResponse:
    """
    Main function to format responses based on available data
    
    Args:
        parsed_action: Enhanced parsed action information
        asset_search_result: Asset search results if available
        trade_recommendation: Trade recommendation if available
        error_message: Error message if something failed
        
    Returns:
        FormattedResponse ready for user
    """
    
    try:
        logger.info(f"📝 Formatting response for intent type: {parsed_action.intent_type.value}")
        
        # Handle errors first
        if error_message:
            return _format_error_response(error_message)
        
        # Handle different intent types
        if parsed_action.intent_type == IntentType.INVALID_QUERY:
            return format_invalid_query_response()
        
        elif parsed_action.intent_type == IntentType.GENERAL_INFO:
            return format_general_info_response(parsed_action)
        
        elif parsed_action.confidence < 0.5:  # Need clarification
            return format_clarification_response(parsed_action)
        
        elif parsed_action.intent_type in [IntentType.TRADE_ANALYSIS, IntentType.MARKET_UPDATE, IntentType.ASSET_COMPARISON]:
            # Handle asset search results
            if not asset_search_result:
                return _format_error_response("No asset search results available")
            
            if not asset_search_result.found:
                return format_asset_not_found_response(asset_search_result)
            
            # Handle trade recommendation
            if not trade_recommendation:
                return _format_error_response("Trade analysis was not completed")
            
            return format_trade_recommendation_response(
                trade_recommendation, 
                asset_search_result.asset_info, 
                parsed_action
            )
        
        elif parsed_action.intent_type in [IntentType.PORTFOLIO_REVIEW, IntentType.RISK_ASSESSMENT]:
            # Future: Handle portfolio and risk management responses
            return format_clarification_response(parsed_action)
        
        else:
            return _format_error_response(f"Unknown intent type: {parsed_action.intent_type.value}")
        
    except Exception as e:
        logger.error(f"❌ Error in main response formatting: {e}")
        return _format_error_response(f"Response formatting failed: {str(e)}")

# --- Convenience Functions ---

def get_response_summary(formatted_response: FormattedResponse) -> str:
    """Get a brief summary of the formatted response"""
    response_type = formatted_response.response_type
    
    if response_type == "trade_recommendation":
        decision = formatted_response.metadata.get("decision", "Unknown")
        symbol = formatted_response.metadata.get("symbol", "Unknown")
        return f"Trade recommendation: {decision} for {symbol}"
    
    elif response_type == "asset_not_found":
        query = formatted_response.metadata.get("failed_query", "Unknown")
        return f"Asset not found: {query}"
    
    elif response_type == "error":
        return "Error response"
    
    else:
        return f"Response type: {response_type}"

def is_actionable_response(formatted_response: FormattedResponse) -> bool:
    """Check if the response contains actionable trading advice"""
    return formatted_response.response_type == "trade_recommendation"

# --- Example Usage ---

if __name__ == "__main__":
    # Example of how the formatter would be used
    # Note: query_parser_agent was removed - would use action_parser instead
    
    # Test different query types
    test_queries = [
        "Should I buy Apple stock?",
        "What is RSI?",
        "How do I cook pasta?",
        "I need more information"
    ]
    
    print("Response Formatter testing would require:")
    print("1. Enhanced Action Parser to create ParsedAction objects")
    print("2. Asset Search Results and Trade Recommendations")
    print("3. Example showing format_response() functionality")
    print(f"\nTest queries available: {test_queries}") 