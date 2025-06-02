"""
Risk Manager Service

Analyzes trade recommendations against user risk tolerance and portfolio context.
Only triggered for actual trade decisions, not general market questions.
"""

from typing import Optional, List, Dict, Any
import logging
from pydantic import BaseModel, Field

from agents.hedge_fund.models import ParsedAction

logger = logging.getLogger(__name__)

class RiskAssessment(BaseModel):
    """Risk assessment result"""
    overall_risk_score: float = Field(ge=0.0, le=1.0, description="Overall risk score 0.0-1.0")
    risk_level: str = Field(description="LOW, MEDIUM, HIGH, VERY_HIGH")
    position_size_recommendation: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    should_proceed: bool = Field(description="Whether to proceed with the trade")
    reasoning: str = Field(description="Risk assessment reasoning")

def assess_trade_risk(
    trade_recommendation: Any,  # TradeRecommendation object
    parsed_action: ParsedAction
) -> RiskAssessment:
    """
    Assess risk of a trade recommendation against user context
    
    Args:
        trade_recommendation: The trade recommendation from technical analysis
        parsed_action: Parsed user query containing risk tolerance and portfolio context
    """
    logger.info(f"🛡️ Assessing risk for {trade_recommendation.symbol} trade")
    
    # Analyze user risk tolerance
    risk_tolerance_score = _assess_risk_tolerance(parsed_action)
    
    # Analyze trade risk factors
    trade_risk_score = _assess_trade_risk_factors(trade_recommendation)
    
    # Analyze portfolio concentration if user has positions
    concentration_risk = _assess_concentration_risk(trade_recommendation, parsed_action)
    
    # Calculate overall risk score (weighted average)
    overall_risk_score = (
        risk_tolerance_score * 0.4 +  # User's risk tolerance
        trade_risk_score * 0.4 +      # Trade-specific risk
        concentration_risk * 0.2       # Portfolio concentration
    )
    
    # Determine risk level
    risk_level = _determine_risk_level(overall_risk_score)
    
    # Generate recommendations and warnings
    warnings, recommendations = _generate_risk_guidance(
        overall_risk_score, risk_level, trade_recommendation, parsed_action
    )
    
    # Determine if trade should proceed
    should_proceed = _should_proceed_with_trade(overall_risk_score, parsed_action)
    
    # Position sizing recommendation
    position_size = _recommend_position_size(
        overall_risk_score, trade_recommendation, parsed_action
    )
    
    reasoning = _generate_risk_reasoning(
        overall_risk_score, risk_tolerance_score, trade_risk_score, 
        concentration_risk, parsed_action
    )
    
    assessment = RiskAssessment(
        overall_risk_score=overall_risk_score,
        risk_level=risk_level,
        position_size_recommendation=position_size,
        warnings=warnings,
        recommendations=recommendations,
        should_proceed=should_proceed,
        reasoning=reasoning
    )
    
    logger.info(f"✅ Risk assessment complete: {risk_level} risk ({overall_risk_score:.2f})")
    return assessment

def _assess_risk_tolerance(parsed_action: ParsedAction) -> float:
    """Assess user's risk tolerance from hints (0.0 = conservative, 1.0 = very aggressive)"""
    
    # Start with risk concern level (inverted - high concern = low tolerance)
    base_score = 1.0 - parsed_action.risk_concern_level
    
    # Adjust based on risk tolerance hints
    if parsed_action.risk_tolerance_hint:
        hint = parsed_action.risk_tolerance_hint.lower()
        if any(word in hint for word in ['conservative', 'safe', 'low risk', 'cautious']):
            base_score = min(base_score, 0.3)
        elif any(word in hint for word in ['very aggressive', 'maximum risk']):
            base_score = max(base_score, 0.9)
        elif any(word in hint for word in ['aggressive', 'high risk', 'risky']):
            base_score = max(base_score, 0.7)
    return base_score

def _assess_trade_risk_factors(trade_recommendation) -> float:
    """Assess inherent risk of the trade recommendation"""
    
    # Base risk from confidence level (lower confidence = higher risk)
    confidence_risk = 1.0 - trade_recommendation.confidence_score
    
    # Assess decision type risk
    decision_risk = 0.5  # Default moderate risk
    if trade_recommendation.decision.value in ["STRONG_BUY", "STRONG_SELL"]:
        decision_risk = 0.7  # Higher risk for strong positions
    elif trade_recommendation.decision.value in ["BUY", "SELL"]:
        decision_risk = 0.5  # Moderate risk
    elif trade_recommendation.decision.value == "HOLD":
        decision_risk = 0.2  # Lower risk
    else:  # NO_TRADE
        decision_risk = 0.1  # Very low risk
    
    # Combine factors
    return (confidence_risk + decision_risk) / 2

def _assess_concentration_risk(trade_recommendation, parsed_action: ParsedAction) -> float:
    """Assess portfolio concentration risk"""
    
    if not parsed_action.existing_positions:
        return 0.3  # Moderate risk for new positions
    
    # Check if user already owns this asset
    symbol = trade_recommendation.symbol
    for position in parsed_action.existing_positions:
        if position.asset.upper() == symbol.upper():
            # User already owns this asset - concentration risk
            if trade_recommendation.decision.value in ["BUY", "STRONG_BUY"]:
                return 0.8  # High concentration risk for buying more
            else:
                return 0.2  # Lower risk for selling
    
    # Count total positions for diversification assessment
    total_positions = len(parsed_action.existing_positions)
    if total_positions < 3:
        return 0.4  # Moderate risk - limited diversification
    elif total_positions > 10:
        return 0.2  # Lower risk - well diversified
    else:
        return 0.3  # Reasonable diversification

def _determine_risk_level(overall_risk_score: float) -> str:
    """Determine risk level category"""
    if overall_risk_score < 0.3:
        return "LOW"
    elif overall_risk_score < 0.5:
        return "MEDIUM"
    elif overall_risk_score < 0.7:
        return "HIGH"
    else:
        return "VERY_HIGH"

def _generate_risk_guidance(
    overall_risk_score: float, 
    risk_level: str, 
    trade_recommendation, 
    parsed_action: ParsedAction
) -> tuple[List[str], List[str]]:
    """Generate warnings and recommendations"""
    
    warnings = []
    recommendations = []
    
    # High risk warnings
    if overall_risk_score > 0.7:
        warnings.append("⚠️ This trade carries very high risk")
        recommendations.append("Consider reducing position size significantly")
    elif overall_risk_score > 0.5:
        warnings.append("⚠️ This trade carries elevated risk")
        recommendations.append("Consider smaller position size")
    
    # User-specific warnings
    if parsed_action.risk_concern_level > 0.6:
        warnings.append("⚠️ You expressed risk concerns - proceed cautiously")
        recommendations.append("Consider paper trading first or very small position")
    
    # Portfolio concentration warnings
    symbol = trade_recommendation.symbol
    for position in parsed_action.existing_positions:
        if position.asset.upper() == symbol.upper():
            if trade_recommendation.decision.value in ["BUY", "STRONG_BUY"]:
                warnings.append(f"⚠️ You already own {position.asset} - concentration risk")
                recommendations.append("Consider diversifying instead of adding to existing position")
    
    # Conservative user guidance
    if parsed_action.risk_tolerance_hint and "conservative" in parsed_action.risk_tolerance_hint.lower():
        recommendations.append("Given your conservative profile, consider dollar-cost averaging")
    
    return warnings, recommendations

def _should_proceed_with_trade(overall_risk_score: float, parsed_action: ParsedAction) -> bool:
    """Determine if trade should proceed based on risk assessment"""
    
    # Very high risk + risk-averse user = don't proceed
    if overall_risk_score > 0.8 and parsed_action.risk_concern_level > 0.6:
        return False
    
    # Extremely high risk = don't proceed regardless
    if overall_risk_score > 0.9:
        return False
    
    return True

def _recommend_position_size(
    overall_risk_score: float, 
    trade_recommendation, 
    parsed_action: ParsedAction
) -> Optional[float]:
    """Recommend position size as percentage of portfolio"""
    
    # Base position size based on risk score
    if overall_risk_score < 0.3:
        base_size = 0.10  # 10% for low risk
    elif overall_risk_score < 0.5:
        base_size = 0.07  # 7% for medium risk
    elif overall_risk_score < 0.7:
        base_size = 0.05  # 5% for high risk
    else:
        base_size = 0.02  # 2% for very high risk
    
    # Adjust for user risk tolerance
    if parsed_action.risk_tolerance_hint:
        hint = parsed_action.risk_tolerance_hint.lower()
        if "conservative" in hint:
            base_size *= 0.5  # Halve for conservative users
        elif "aggressive" in hint:
            base_size *= 1.5  # Increase for aggressive users
    
    # Adjust for confidence
    confidence_multiplier = trade_recommendation.confidence_score
    adjusted_size = base_size * confidence_multiplier
    
    return max(0.01, min(0.15, adjusted_size))  # Cap between 1% and 15%

def _generate_risk_reasoning(
    overall_risk_score: float,
    risk_tolerance_score: float, 
    trade_risk_score: float,
    concentration_risk: float,
    parsed_action: ParsedAction
) -> str:
    """Generate detailed risk reasoning"""
    
    reasoning_parts = [
        f"Risk Assessment: Overall risk score {overall_risk_score:.2f}/1.0"
    ]
    
    # Risk tolerance component
    if risk_tolerance_score < 0.4:
        reasoning_parts.append(f"Conservative risk profile detected (score: {risk_tolerance_score:.2f})")
    elif risk_tolerance_score > 0.7:
        reasoning_parts.append(f"Aggressive risk profile detected (score: {risk_tolerance_score:.2f})")
    else:
        reasoning_parts.append(f"Moderate risk profile (score: {risk_tolerance_score:.2f})")
    
    # Trade risk component
    if trade_risk_score > 0.6:
        reasoning_parts.append(f"High trade-specific risk (score: {trade_risk_score:.2f})")
    else:
        reasoning_parts.append(f"Moderate trade risk (score: {trade_risk_score:.2f})")
    
    # Concentration risk
    if concentration_risk > 0.6:
        reasoning_parts.append(f"Elevated concentration risk (score: {concentration_risk:.2f})")
    
    # User concerns
    if parsed_action.risk_concern_level > 0.5:
        reasoning_parts.append("User expressed risk concerns in query")
    
    return ". ".join(reasoning_parts) + "." 