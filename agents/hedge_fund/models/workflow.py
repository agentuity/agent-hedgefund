"""
Workflow Types

Contains types related to the LangGraph workflow state and routing.
"""

from typing import Dict, Optional, Any, TypedDict
from enum import Enum

class NextAction(Enum):
    """
    Enum for next action routing in the workflow
    Prevents spelling mistakes and provides type safety
    """
    # From parsing
    REJECT_QUERY = "reject_query"
    GENERAL_RESPONSE = "general_response"
    CLARIFICATION = "clarification" 
    SEARCH_ASSET = "search_asset"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    RISK_ANALYSIS = "risk_analysis"
    
    # From asset search
    ANALYZE_TRADE = "analyze_trade"
    SUGGEST_ALTERNATIVES = "suggest_alternatives"
    
    # From analysis
    ASSESS_RISK = "assess_risk"
    ANALYZE_PORTFOLIO = "analyze_portfolio"
    FORMAT_RESPONSE = "format_response"
    
    # Error handling
    FORMAT_ERROR = "format_error"

class HedgeFundState(TypedDict):
    """
    State schema for the hedge fund analysis workflow
    Contains all data that flows through the LangGraph nodes
    """
    # Input
    user_query: str
    user_context: Optional[Dict[str, Any]]
    
    # Parsed information (from action parser)
    parsed_action: Optional[Any]  # Will be ParsedAction once imported
    
    # Asset search results
    asset_search_result: Optional[Any]  # Will be AssetSearchResult once imported
    is_valid_asset: bool
    
    # Analysis results  
    trade_recommendation: Optional[Any]  # Will be TradeRecommendation once imported
    
    # Risk assessment results
    risk_assessment: Optional[Any]  # Will be RiskAssessment once imported
    
    # Final output
    formatted_response: Optional[Any]  # Will be FormattedResponse once imported
    
    # Workflow control
    next_action: Optional[NextAction]
    response_ready: bool
    error: Optional[str]
    debug_info: Optional[Dict[str, Any]]
    confidence_score: Optional[float] 