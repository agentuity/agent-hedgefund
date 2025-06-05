"""
LLM Response Generation Node

Uses an LLM to synthesize all analysis data into natural, conversational responses.
Replaces complex string-building logic with intelligent response generation.
"""

import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agents.hedge_fund.models import NextAction, HedgeFundState, IntentType
from agents.hedge_fund.services.response_formatter_service import FormattedResponse

logger = logging.getLogger(__name__)

def generate_llm_response_node(state: HedgeFundState) -> HedgeFundState:
    """Generate response using LLM to synthesize all analysis data"""
    try:
        parsed_action = state.get("parsed_action")
        asset_search_result = state.get("asset_search_result")
        trade_recommendation = state.get("trade_recommendation")
        risk_assessment = state.get("risk_assessment")
        portfolio_decision = state.get("portfolio_decision")
        
        if not parsed_action:
            state["error"] = "No parsed action available for response generation"
            state["next_action"] = NextAction.FORMAT_ERROR
            return state
        
        logger.info(f"🤖 Generating LLM response for: {parsed_action.original_query}")
        
        # Check if we have a concrete portfolio decision to format
        if portfolio_decision:
            logger.info("📋 Formatting concrete portfolio decision into trade table")
            response_content = _format_portfolio_decision_table(
                portfolio_decision, asset_search_result, trade_recommendation, risk_assessment
            )
        else:
            logger.info("💭 No portfolio decision found - generating general LLM response")
            # Create comprehensive prompt with all analysis data
            prompt = _create_response_prompt(
                parsed_action, asset_search_result, trade_recommendation, risk_assessment
            )
            
            # Generate response using LLM
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            response = llm.invoke([HumanMessage(content=prompt)])
            response_content = response.content
        
        # Create formatted response
        formatted_response = FormattedResponse(
            content=response_content,
            response_type="portfolio_decision" if portfolio_decision else "llm_generated",
            metadata={
                "original_query": parsed_action.original_query,
                "intent_type": parsed_action.intent_type.value,
                "has_trade_recommendation": trade_recommendation is not None,
                "has_risk_assessment": risk_assessment is not None,
                "has_portfolio_decision": portfolio_decision is not None,
                "trade_intent_strength": parsed_action.trade_intent_strength,
                "confidence": parsed_action.confidence
            },
            timestamp=datetime.now()
        )
        
        state["formatted_response"] = formatted_response
        state["response_ready"] = True
        
        logger.info("✅ LLM response generated successfully")
        
    except Exception as e:
        state["error"] = f"LLM response generation failed: {str(e)}"
        state["next_action"] = NextAction.FORMAT_ERROR
        logger.error(f"❌ LLM response generation error: {e}")
    
    return state

def _format_portfolio_decision_table(portfolio_decision, asset_search_result, trade_recommendation, risk_assessment) -> str:
    """Format portfolio decision into concrete trade table"""
    
    decision = portfolio_decision.decision
    symbol = decision.symbol
    action = decision.action.value.upper()
    quantity = decision.quantity
    
    # Get price info
    current_price = 0.0
    if asset_search_result and asset_search_result.asset_info:
        current_price = asset_search_result.asset_info.current_price or 0.0
    
    # Calculate derived values
    total_amount = quantity * current_price if current_price > 0 else decision.amount_usd
    stop_loss = current_price * 0.9 if current_price > 0 else 0  # 10% stop loss
    take_profit = current_price * 1.1 if current_price > 0 else 0  # 10% take profit
    
    # Get risk info
    risk_level = "MEDIUM"
    if risk_assessment:
        risk_level = risk_assessment.risk_level.upper()
    
    # Format the response based on action
    if action == "BUY":
        response = f"""🎯 TRADE RECOMMENDATION: {action}

┌─────────────────┬──────────────────┐
│ Symbol          │ {symbol:<16} │
│ Action          │ {action:<16} │
│ Quantity        │ {quantity} shares{'':<7} │
│ Entry Price     │ ${current_price:.2f}{'':<11} │
│ Total Cost      │ ${total_amount:,.2f}{'':<8} │
│ Stop Loss       │ ${stop_loss:.2f} (-10%){'':<5} │
│ Take Profit     │ ${take_profit:.2f} (+10%){'':<5} │
│ Risk Level      │ {risk_level:<16} │
│ Confidence      │ {decision.confidence:.0f}%{'':<12} │
└─────────────────┴──────────────────┘

💡 **Action Required:**
• Place market buy order for {quantity} {symbol} shares
• Set stop loss at ${stop_loss:.2f}
• Monitor position for exit signals

💭 **Reasoning:** {decision.reasoning}

📊 **Portfolio Impact:**
• Cash allocation after trade: {portfolio_decision.cash_allocation_after_trade:.1f}%
• Expected outcome: {portfolio_decision.expected_outcome}
• Overall strategy: {portfolio_decision.overall_strategy}"""

    elif action == "SELL":
        response = f"""🎯 TRADE RECOMMENDATION: {action}

┌─────────────────┬──────────────────┐
│ Symbol          │ {symbol:<16} │
│ Action          │ {action:<16} │
│ Quantity        │ {quantity} shares{'':<7} │
│ Current Price   │ ${current_price:.2f}{'':<11} │
│ Total Value     │ ${total_amount:,.2f}{'':<8} │
│ Exit Reason     │ Analysis signal{'':<2} │
│ Urgency         │ NORMAL{'':<10} │
└─────────────────┴──────────────────┘

💡 **Action Required:**
• Place market sell order for {quantity} {symbol} shares
• Expected proceeds: ${total_amount:,.2f}

💭 **Reasoning:** {decision.reasoning}

📊 **Portfolio Impact:**
• Cash allocation after trade: {portfolio_decision.cash_allocation_after_trade:.1f}%
• Expected outcome: {portfolio_decision.expected_outcome}"""

    else:  # HOLD
        response = f"""🎯 TRADE RECOMMENDATION: {action}

┌─────────────────┬──────────────────┐
│ Symbol          │ {symbol:<16} │
│ Action          │ {action:<16} │
│ Current Price   │ ${current_price:.2f}{'':<11} │
│ Reason          │ Mixed signals{'':<3} │
│ Next Review     │ In 1 week{'':<7} │
└─────────────────┴──────────────────┘

💡 **Action Required:**
• No immediate action needed
• Monitor for clearer signals

💭 **Reasoning:** {decision.reasoning}"""

    return response

def _create_response_prompt(
    parsed_action,
    asset_search_result=None,
    trade_recommendation=None,
    risk_assessment=None
) -> str:
    """Create comprehensive prompt for LLM response generation"""
    
    if(parsed_action.intent_type == IntentType.INVALID_QUERY):
        return "I'm sorry, I can only help with trading and investment questions. Please ask about a specific stock or crypto asset."
    elif(parsed_action.intent_type == IntentType.UNCLEAR):
        return "I'm sorry, I'm not sure what you're asking. Please ask about a specific stock or crypto asset."
    
    prompt_parts = [
        "You are a professional hedge fund trading assistant. Your job is to provide clear, direct, and actionable responses to trading questions.",
        "",
        "TASK: Answer the user's question based on the analysis data provided below.",
        "If the user's question is not related to trading or investment, please respond with a helpful message and keep it precise.",
        "",
        f"USER'S ORIGINAL QUESTION: \"{parsed_action.original_query}\"",
        "",
        "ANALYSIS DATA:",
    ]
    
    # Add user context information
    prompt_parts.extend([
        "",
        "USER CONTEXT:",
        f"- Trade Intent Strength: {parsed_action.trade_intent_strength:.1%}",
        f"- Risk Concern Level: {parsed_action.risk_concern_level:.1%}",
        f"- Portfolio Context Strength: {parsed_action.portfolio_context_strength:.1%}",
    ])
    
    if parsed_action.risk_tolerance_hint:
        prompt_parts.append(f"- Risk Tolerance: {parsed_action.risk_tolerance_hint}")
    
    if parsed_action.existing_positions:
        positions = [f"{pos.asset} ({pos.quantity})" for pos in parsed_action.existing_positions]
        prompt_parts.append(f"- Existing Positions: {', '.join(positions)}")
    
    # Add asset information if available
    if asset_search_result and asset_search_result.found:
        asset = asset_search_result.asset_info
        prompt_parts.extend([
            "",
            "ASSET INFORMATION:",
            f"- Name: {asset.name} ({asset.symbol})",
            f"- Type: {asset.asset_type.value}",
            f"- Current Price: ${asset.current_price:.2f}" if asset.current_price else "- Current Price: Not available",
            f"- Exchange: {asset.exchange}" if asset.exchange else ""
        ])
    
    # Add trade recommendation if available
    if trade_recommendation:
        prompt_parts.extend([
            "",
            "TECHNICAL ANALYSIS:",
            f"- Decision: {trade_recommendation.decision.value}",
            f"- Confidence: {trade_recommendation.confidence.value} ({trade_recommendation.confidence_score:.1%})",
            f"- Market Context: {trade_recommendation.market_context}",
            f"- Analysis: {trade_recommendation.reasoning}",
            "",
            "KEY FACTORS:",
        ])
        
        for i, factor in enumerate(trade_recommendation.key_factors[:3], 1):
            prompt_parts.append(f"  {i}. {factor}")
        
        # Add enhanced insights if available
        if trade_recommendation.enhanced_insights:
            insights = trade_recommendation.enhanced_insights
            prompt_parts.append("")
            prompt_parts.append("ENHANCED INSIGHTS:")
            
            if insights.get('key_themes'):
                themes = ', '.join(insights['key_themes'][:3])
                prompt_parts.append(f"- Market Themes: {themes}")
            
            if insights.get('market_moving_events'):
                events = ', '.join(insights['market_moving_events'][:2])
                prompt_parts.append(f"- Key Events: {events}")
            
            if insights.get('risk_factors'):
                risks = ', '.join(insights['risk_factors'][:2])
                prompt_parts.append(f"- Risk Factors: {risks}")
    
    # Add risk assessment if available
    if risk_assessment:
        prompt_parts.extend([
            "",
            "RISK ASSESSMENT:",
            f"- Risk Level: {risk_assessment.risk_level}",
            f"- Risk Score: {risk_assessment.overall_risk_score:.2f}/1.0",
            f"- Recommended Position Size: {risk_assessment.position_size_recommendation:.1%} of portfolio",
            f"- Should Proceed: {risk_assessment.should_proceed}",
            f"- Risk Analysis: {risk_assessment.reasoning}",
        ])
        
        if risk_assessment.warnings:
            prompt_parts.append("")
            prompt_parts.append("WARNINGS:")
            for warning in risk_assessment.warnings:
                prompt_parts.append(f"  - {warning}")
        
        if risk_assessment.recommendations:
            prompt_parts.append("")
            prompt_parts.append("RISK RECOMMENDATIONS:")
            for rec in risk_assessment.recommendations:
                prompt_parts.append(f"  - {rec}")
    
    # Add response guidelines
    prompt_parts.extend([
        "",
        "RESPONSE GUIDELINES:",
        "1. Start with a DIRECT answer to their specific question (Yes/No/Maybe + brief reasoning)",
        "2. Be conversational and professional, not robotic",
        "3. Highlight the most important insights from the analysis",
        "4. If this is a trading decision, be clear about your recommendation",
        "5. Address any risk concerns they mentioned",
        "6. Keep the response concise but complete (aim for 150-300 words)",
        "7. Use appropriate emojis sparingly for readability",
        "8. End with a clear summary of your recommendation",
        "",
        "IMPORTANT:",
        "- If they asked about buying/selling, give a clear yes/no recommendation",
        "- If they have existing positions mentioned, address concentration risk",
        "- If risk assessment shows high risk, emphasize caution",
        "- If they seem risk-averse, emphasize safety factors",
        "- If they seem aggressive, you can be more bullish in tone",
        "",
        "Generate your response now:"
    ])
    
    return "\n".join(prompt_parts)

def format_error_node(state: HedgeFundState) -> HedgeFundState:
    """Format error responses using LLM for natural error messages"""
    try:
        error_message = state.get("error", "Unknown error occurred")
        user_query = state.get("user_query", "")
        
        logger.info(f"📝 Generating LLM error response: {error_message}")
        
        # Create error response prompt
        prompt = f"""You are a professional trading assistant. The user asked: "{user_query}"

Unfortunately, an error occurred: {error_message}

Please provide a helpful, professional response that:
1. Acknowledges their question
2. Explains what went wrong in simple terms
3. Suggests what they could try instead
4. Maintains a helpful, professional tone

Keep it brief and actionable."""
        
        # Generate response using LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        response = llm.invoke([HumanMessage(content=prompt)])
        
        formatted_response = FormattedResponse(
            content=response.content,
            response_type="error",
            metadata={
                "error_message": error_message,
                "original_query": user_query,
                "llm_generated": True
            }
        )
        
        state["formatted_response"] = formatted_response
        state["response_ready"] = True
        
        logger.info("✅ LLM error response generated")
        
    except Exception as e:
        # Fallback error handling
        logger.error(f"❌ Error generating LLM error response: {e}")
        state["formatted_response"] = FormattedResponse(
            content="Sorry, I encountered an unexpected error. Please try asking about a specific stock or crypto asset.",
            response_type="error",
            metadata={"fallback_error": True}
        )
        state["response_ready"] = True
    
    return state 