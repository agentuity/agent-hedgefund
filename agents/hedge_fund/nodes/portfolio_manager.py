"""
Portfolio Manager Node

Makes final BUY/SELL decisions with specific quantities based on:
- Technical/sentiment analysis results
- User's current position (if any)
- Available cash and risk limits
- Bullish vs bearish signals
"""

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agents.hedge_fund.models.workflow import HedgeFundState, NextAction
from agents.hedge_fund.models.portfolio_decision import (
    PortfolioDecision, 
    PortfolioManagerOutput, 
    TradeAction
)

logger = logging.getLogger(__name__)

def portfolio_manager_node(state: HedgeFundState) -> HedgeFundState:
    """
    Portfolio Manager: Makes final BUY/SELL decisions with quantities
    
    Role: Given bullish/bearish analysis, decide specific action and quantity
    """
    
    logger.info("💼 Portfolio manager starting final decision process")
    
    try:
        # Extract required data
        asset_search_result = state.get("asset_search_result")
        trade_recommendation = state.get("trade_recommendation")
        risk_assessment = state.get("risk_assessment")
        parsed_action = state.get("parsed_action")
        
        if not asset_search_result or not trade_recommendation:
            logger.error("❌ Missing required data for portfolio decision")
            return {
                **state,
                "error": "Missing required data for portfolio decision",
                "next_action": NextAction.FORMAT_ERROR
            }
        
        # Validate asset search result structure
        if not asset_search_result.found or not asset_search_result.asset_info:
            logger.error("❌ Invalid asset search result for portfolio decision")
            return {
                **state,
                "error": "Invalid asset search result for portfolio decision",
                "next_action": NextAction.FORMAT_ERROR
            }
        
        # Get basic info safely
        asset_info = asset_search_result.asset_info
        symbol = asset_info.symbol
        logger.info(f"💼 Making portfolio decision for {symbol}")
        
        # Handle potential None current_price
        current_price = asset_info.current_price
        if current_price is None or current_price <= 0:
            # Fallback to a reasonable default or try to get from trade recommendation
            current_price = getattr(trade_recommendation, 'current_price', 100.0)
            if current_price is None or current_price <= 0:
                current_price = 100.0  # Last resort fallback
        
        logger.info(f"💰 Current price: ${current_price:.2f}")
        
        # Get user's current position (if any)
        current_position = _get_current_position(parsed_action, symbol)
        logger.info(f"📊 Current position: {current_position['description']}")
        
        # Calculate position sizing
        position_info = _calculate_simple_position_sizing(
            current_price=current_price,
            current_position=current_position,
            trade_recommendation=trade_recommendation,
            risk_assessment=risk_assessment
        )
        
        logger.info(f"💡 Position sizing: {position_info['recommended_shares']} shares (~${position_info['recommended_investment']:.0f})")
        
        # Generate final decision
        logger.info("🤖 Generating final portfolio decision with LLM...")
        portfolio_decision = _make_portfolio_decision(
            symbol=symbol,
            current_price=current_price,
            current_position=current_position,
            position_info=position_info,
            trade_recommendation=trade_recommendation,
            risk_assessment=risk_assessment,
            parsed_action=parsed_action
        )
        
        logger.info(f"✅ Portfolio decision: {portfolio_decision.decision.action.value} {portfolio_decision.decision.quantity} shares")
        logger.info("💼 Portfolio manager complete - routing to response generation")
        
        # Store decision and proceed to response formatting
        return {
            **state,
            "portfolio_decision": portfolio_decision,
            "next_action": NextAction.FORMAT_RESPONSE,
            "response_ready": True
        }
        
    except Exception as e:
        logger.error(f"❌ Portfolio decision error: {str(e)}")
        return {
            **state,
            "error": f"Portfolio decision error: {str(e)}",
            "next_action": NextAction.FORMAT_ERROR
        }

def _get_current_position(parsed_action, symbol: str) -> dict:
    """Extract current position info from user query"""
    current_position = {
        "has_position": False,
        "quantity": 0,
        "description": "No current position"
    }
    
    if not parsed_action or not hasattr(parsed_action, 'existing_positions'):
        return current_position
    
    # Look for mentions of current position
    for position in parsed_action.existing_positions:
        if symbol.upper() in position.get("asset", "").upper():
            current_position = {
                "has_position": True,
                "quantity": position.get("quantity", 0),
                "description": f"Currently own {position.get('quantity', 0)} shares"
            }
            break
    
    return current_position

def _calculate_simple_position_sizing(
    current_price: float,
    current_position: dict,
    trade_recommendation,
    risk_assessment
) -> dict:
    """Simple position sizing based on available cash and risk"""
    
    # Default assumptions for position sizing
    available_cash = 10000.0  # Default assumption
    max_position_percentage = 10.0  # Max 10% of portfolio per position
    portfolio_value = 50000.0  # Default portfolio assumption
    
    # Calculate max investment amount
    max_investment = min(
        available_cash * 0.8,  # Keep 20% cash buffer
        portfolio_value * (max_position_percentage / 100)
    )
    
    # Risk adjustment based on analysis
    risk_multiplier = 1.0
    if risk_assessment and hasattr(risk_assessment, 'risk_level'):
        if risk_assessment.risk_level.upper() == "HIGH":
            risk_multiplier = 0.5
        elif risk_assessment.risk_level.upper() == "MEDIUM":
            risk_multiplier = 0.75
    
    # Adjust for recommendation strength
    if trade_recommendation and hasattr(trade_recommendation, 'confidence'):
        confidence_obj = getattr(trade_recommendation, "confidence", "")
        confidence_str = getattr(confidence_obj, "value", str(confidence_obj))
        if "LOW" in confidence_str.upper():
            risk_multiplier *= 0.6
        elif "HIGH" in confidence_str.upper():
            risk_multiplier *= 1.2
    
    # Final investment amount
    recommended_investment = max_investment * risk_multiplier
    recommended_shares = int(recommended_investment / current_price) if current_price > 0 else 0
    
    return {
        "max_investment": max_investment,
        "recommended_investment": recommended_investment,
        "recommended_shares": recommended_shares,
        "risk_multiplier": risk_multiplier
    }

def _make_portfolio_decision(
    symbol: str,
    current_price: float,
    current_position: dict,
    position_info: dict,
    trade_recommendation,
    risk_assessment,
    parsed_action
) -> PortfolioManagerOutput:
    """Make final BUY/SELL decision using LLM"""
    
    # Prepare recommendation summary
    recommendation_summary = _get_recommendation_summary(trade_recommendation)
    risk_summary = _get_risk_summary(risk_assessment)
    
    # Simple LLM prompt for BUY/SELL decision
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a portfolio manager making BUY/SELL decisions.

Based on the analysis, decide:
1. Action: "buy", "sell", or "hold" 
2. Quantity: specific number of shares
3. Reasoning: why this decision makes sense

Rules:
- BUY: Only if bullish signals AND have cash available
- SELL: Only if bearish signals AND currently own shares  
- HOLD: If unclear signals or no cash/position

Output JSON exactly as shown (ALL fields required):
{{
    "decision": {{
        "symbol": "{symbol}",
        "action": "buy/sell/hold",
        "quantity": number,
        "amount_usd": dollar_amount,
        "confidence": percentage,
        "reasoning": "clear reasoning"
    }},
    "overall_strategy": "brief strategy description",
    "risk_assessment": "portfolio risk evaluation after this trade",
    "cash_allocation_after_trade": percentage,
    "expected_outcome": "expected result from this trade",
    "confidence_score": percentage
}}"""),
        
        ("human", """Make your decision for {symbol} at ${current_price:.2f}

CURRENT POSITION: {current_position_desc}

ANALYSIS:
{recommendation_summary}
{risk_summary}

POSITION SIZING:
- Can invest up to: ${max_investment:.2f}
- Recommended investment: ${recommended_investment:.2f}  
- Recommended shares: {recommended_shares}
- Risk adjustment: {risk_multiplier:.2f}x

USER QUERY: {user_query}

Decision (include ALL required fields):""")
    ])
    
    # Generate prompt
    prompt = template.invoke({
        "symbol": symbol,
        "current_price": current_price,
        "current_position_desc": current_position["description"],
        "recommendation_summary": recommendation_summary,
        "risk_summary": risk_summary,
        "max_investment": position_info["max_investment"],
        "recommended_investment": position_info["recommended_investment"],
        "recommended_shares": position_info["recommended_shares"],
        "risk_multiplier": position_info["risk_multiplier"],
        "user_query": getattr(parsed_action, 'original_query', "N/A")
    })
    
    # Get LLM decision
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        response = llm.invoke(prompt)
        decision_data = json.loads(response.content)
        return PortfolioManagerOutput(**decision_data)
        
    except Exception as e:
        # Safe fallback with ALL required fields
        fallback_decision = PortfolioDecision(
            symbol=symbol,
            action=TradeAction.HOLD,
            quantity=0,
            confidence=0.0,
            amount_usd=0.0,
            reasoning=f"Error in decision making: {str(e)}. Holding for safety."
        )
        
        return PortfolioManagerOutput(
            decision=fallback_decision,
            overall_strategy="Conservative hold due to error",
            risk_assessment="Unable to assess portfolio risk due to error",
            cash_allocation_after_trade=100.0,
            expected_outcome="No action taken due to error",
            confidence_score=0.0
        )

def _get_recommendation_summary(trade_recommendation) -> str:
    """Get summary of trade recommendation"""
    if not trade_recommendation:
        return "No trade recommendation available"
    
    parts = []
    if hasattr(trade_recommendation, 'decision'):
        parts.append(f"Recommendation: {trade_recommendation.decision.value}")
    if hasattr(trade_recommendation, 'confidence'):
        parts.append(f"Confidence: {trade_recommendation.confidence.value}")
    if hasattr(trade_recommendation, 'key_signals'):
        signals = getattr(trade_recommendation, 'key_signals', [])
        if signals:
            parts.append(f"Key Signals: {', '.join(signals)}")
    
    return "\n".join(parts) if parts else "Basic recommendation available"

def _get_risk_summary(risk_assessment) -> str:
    """Get summary of risk assessment"""
    if not risk_assessment:
        return "No risk assessment available"
    
    parts = []
    if hasattr(risk_assessment, 'risk_level'):
        parts.append(f"Risk Level: {risk_assessment.risk_level}")
    if hasattr(risk_assessment, 'key_risks'):
        risks = getattr(risk_assessment, 'key_risks', [])
        if risks:
            parts.append(f"Key Risks: {', '.join(risks)}")
    
    return "\n".join(parts) if parts else "Basic risk assessment available" 