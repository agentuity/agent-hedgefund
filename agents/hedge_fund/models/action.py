"""
Action Types

Contains types related to query parsing and action extraction.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

class IntentType(Enum):
    """User's primary intent"""
    STOCK_ANALYSIS = "stock_analysis"
    CRYPTO_ANALYSIS = "crypto_analysis" 
    PORTFOLIO_REVIEW = "portfolio_review"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_EXECUTION = "trade_execution"
    EDUCATION = "education"
    
    # Additional types expected by the codebase
    TRADE_ANALYSIS = "trade_analysis"
    MARKET_UPDATE = "market_update"
    ASSET_COMPARISON = "asset_comparison"
    GENERAL_INFO = "general_info"
    INVALID_QUERY = "invalid_query"
    
    UNCLEAR = "unclear"

class TradeDirection(Enum):
    """Direction of intended trade"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    UNCLEAR = "unclear"

class PositionMention(BaseModel):
    """Information about existing positions mentioned by user"""
    asset: str = Field(description="Asset symbol or name")
    quantity: Optional[float] = Field(default=None, description="Number of shares/units owned")
    entry_price: Optional[float] = Field(default=None, description="Price when position was entered")
    current_value: Optional[float] = Field(default=None, description="Current value of position")
    
class QuantityMention(BaseModel):
    """Quantity or sizing information from user"""
    amount: Optional[float] = Field(default=None, description="Numerical amount mentioned")
    unit: Optional[str] = Field(default=None, description="Unit type: shares, dollars, percent, etc.")
    is_percentage: bool = Field(default=False, description="Whether the amount is a percentage")

class ParsedAction(BaseModel):
    """
    Comprehensive extraction of user intent and context from financial queries.
    This represents WHAT the user wants (pure information extraction).
    """
    
    # Core Intent
    intent_type: IntentType = Field(
        description="Primary financial intent: trade_analysis for buy/sell decisions, portfolio_review for portfolio questions, risk_assessment for risk evaluation, market_update for current market status, general_info for educational questions, invalid_query for non-financial topics"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in intent classification (0.0-1.0). Higher when intent is clear."
    )
    
    # Primary Asset Analysis
    primary_asset: Optional[str] = Field(
        default=None,
        description="Main asset mentioned: company name (Tesla, Apple), stock symbol (AAPL, TSLA), or crypto (Bitcoin, BTC-USD). Extract the most recognizable form."
    )
    asset_type_hint: Optional[str] = Field(
        default=None,
        description="Type hint: 'stock', 'crypto', 'etf', etc. based on context clues"
    )
    
    # Trade Context
    trade_direction: TradeDirection = Field(
        default=TradeDirection.UNCLEAR,
        description="Intended trade direction: buy for purchase intent, sell for selling intent, hold for holding analysis, unclear if not specified"
    )
    quantity_info: Optional[QuantityMention] = Field(
        default=None,
        description="Quantity or monetary amount mentioned in the query"
    )
    
    # Portfolio Context
    existing_positions: List[PositionMention] = Field(
        default_factory=list,
        description="Any existing positions mentioned by user (e.g., 'I own 100 shares of Tesla')"
    )
    portfolio_context: bool = Field(
        default=False,
        description="True if user mentions existing holdings or portfolio composition"
    )
    
    # Risk Context  
    risk_mentioned: bool = Field(
        default=False,
        description="True if user mentions risk, volatility, safety, or risk-related concerns"
    )
    risk_tolerance_hint: Optional[str] = Field(
        default=None,
        description="Risk tolerance hints: 'conservative', 'aggressive', 'moderate', etc."
    )
    time_horizon_hint: Optional[str] = Field(
        default=None,
        description="Investment timeframe: 'short-term', 'long-term', 'day-trading', etc."
    )
    
    # Signal Strength Indicators (0.0-1.0)
    urgency_level: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Urgency level: 1.0 for 'now/today/urgent', 0.5 for normal timing, 0.0 for no time pressure"
    )
    specificity_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How specific the request is: 1.0 for very specific, 0.0 for very vague"
    )
    trade_intent_strength: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Strength of trading intent: 1.0 for 'should I buy', 0.5 for 'tell me about', 0.0 for general questions"
    )
    portfolio_context_strength: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Amount of portfolio context provided: 1.0 if mentions specific positions, 0.5 for general portfolio questions, 0.0 for no portfolio context"
    )
    risk_concern_level: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Level of risk concern expressed: 1.0 for explicit risk questions, 0.5 for risk keywords, 0.0 for no risk concern"
    )
    
    # Extracted Context
    key_concerns: List[str] = Field(
        default_factory=list,
        description="Main topics or concerns mentioned by the user"
    )
    mentioned_indicators: List[str] = Field(
        default_factory=list,
        description="Technical indicators mentioned (RSI, MACD, SMA, etc.)"
    )
    
    # Legacy properties for compatibility
    additional_assets: List[str] = Field(
        default_factory=list,
        description="Other assets mentioned beyond the primary asset"
    )
    mentioned_positions: List[PositionMention] = Field(
        default_factory=list,
        description="Alias for existing_positions for backward compatibility"
    )
    quantities: List[QuantityMention] = Field(
        default_factory=list,
        description="All quantity mentions found in the query"
    )
    time_references: List[str] = Field(
        default_factory=list,
        description="Time-related words: 'today', 'tomorrow', 'next week', etc."
    )
    risk_keywords: List[str] = Field(
        default_factory=list,
        description="Risk-related words found: 'risky', 'safe', 'volatile', etc."
    )
    market_events: List[str] = Field(
        default_factory=list,
        description="Market events mentioned: 'earnings', 'fed meeting', 'economic data', etc."
    )
    additional_metadata: Optional[str] = Field(
        default=None,
        description="Additional context or metadata as a JSON string if needed"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the parsing decision"
    )
    
    # Fallback
    original_query: str = Field(
        description="The exact original user query for reference"
    )
    requires_clarification: bool = Field(
        default=False,
        description="True if the query is too ambiguous and needs clarification"
    )
    clarification_needed: List[str] = Field(
        default_factory=list,
        description="Specific aspects that need clarification from the user"
    ) 