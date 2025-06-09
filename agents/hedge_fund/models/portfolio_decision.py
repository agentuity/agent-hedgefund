"""
Portfolio Decision Models

Defines the data structures for portfolio management decisions and trade execution.
Simplified to support only basic long positions: HOLD, BUY, and SELL.
"""

from typing import Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class TradeAction(str, Enum):
    """Available trading actions - simplified to basic long positions"""
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"

class PortfolioDecision(BaseModel):
    """Individual trading decision for a specific asset"""
    symbol: str = Field(description="Asset symbol (e.g., AAPL, BTC)")
    action: TradeAction = Field(description="Trading action to take")
    quantity: int = Field(default=0, ge=0, description="Number of shares/units to trade")
    amount_usd: Optional[float] = Field(default=None, ge=0, description="Dollar amount to invest/divest")
    confidence: float = Field(ge=0.0, le=100.0, description="Confidence in the decision (0-100%)")
    reasoning: str = Field(description="Detailed reasoning for the decision")
    
    # Risk considerations
    position_size_percentage: Optional[float] = Field(default=None, description="Percentage of portfolio this position represents")
    stop_loss_price: Optional[float] = Field(default=None, description="Suggested stop loss price")
    
    # Timing considerations
    urgency: str = Field(default="normal", description="Urgency of trade execution: immediate, normal, or patient")

class PortfolioManagerOutput(BaseModel):
    """Complete output from portfolio management analysis"""
    decision: PortfolioDecision = Field(description="Trading decision for the requested asset")
    
    # Portfolio-level insights
    overall_strategy: str = Field(description="Overall portfolio strategy being implemented")
    risk_assessment: str = Field(description="Portfolio-level risk assessment")
    cash_allocation_after_trade: float = Field(description="Cash allocation percentage after trade")
    
    # Performance expectations
    expected_outcome: str = Field(description="Expected outcome from this trade")
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(ge=0.0, le=100.0, description="Overall confidence in the decision")

class PositionSizing(BaseModel):
    """Position sizing calculations and constraints"""
    symbol: str
    current_price: float
    max_position_value: float
    recommended_shares: int
    recommended_value: float
    risk_adjusted_shares: int
    position_limit_reason: str
    
class PortfolioContext(BaseModel):
    """Context information for portfolio decision making"""
    total_portfolio_value: float
    available_cash: float
    current_positions: Dict[str, Dict] = Field(default_factory=dict)
    sector_exposures: Dict[str, float] = Field(default_factory=dict)
    risk_limits: Dict[str, float] = Field(default_factory=dict) 