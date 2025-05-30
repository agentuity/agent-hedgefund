"""
Trade service for the Hedge Fund Agent
Handles trade analysis coordination and business logic
"""

from agents.hedge_fund.agents.trade_decision_agent import (
    TradeAnalysisRequest, 
    TradeRecommendation,
    run_trade_decision_analysis
)
from agents.hedge_fund.tools import AssetInfo, TradeIntent
from agents.hedge_fund.utils.type_converters import convert_asset_type


class TradeService:
    """
    Service responsible for coordinating trade analysis
    Encapsulates trade decision logic and analysis workflow
    """
    
    def analyze_trade(self, asset_info: AssetInfo, trade_intent: TradeIntent, timeframe: str = "1d") -> TradeRecommendation:
        """
        Perform comprehensive trade analysis for an asset
        
        Args:
            asset_info: Information about the asset to analyze
            trade_intent: User's intended trade action
            timeframe: Analysis timeframe (default: "1d")
            
        Returns:
            TradeRecommendation with analysis results
            
        Raises:
            Exception: If trade analysis fails
        """
        if not asset_info or not asset_info.symbol:
            raise ValueError("Invalid asset information provided for trade analysis")
        
        # Convert asset type for trade decision system
        trade_asset_type = convert_asset_type(asset_info.asset_type)
        
        # Create trade analysis request
        trade_request = TradeAnalysisRequest(
            symbol=asset_info.symbol,
            asset_type=trade_asset_type,
            timeframe=timeframe
        )
        
        # Run the trade decision analysis
        try:
            recommendation = run_trade_decision_analysis(trade_request)
            return recommendation
        except Exception as e:
            raise Exception(f"Trade analysis failed for {asset_info.symbol}: {str(e)}") from e
    
    def validate_trade_request(self, asset_info: AssetInfo, trade_intent: TradeIntent) -> bool:
        """
        Validate if a trade request can be processed
        
        Args:
            asset_info: Asset information to validate
            trade_intent: User's trade intent
            
        Returns:
            True if request is valid, False otherwise
        """
        if not asset_info:
            return False
        
        if not asset_info.symbol or not asset_info.asset_type:
            return False
        
        if trade_intent is None:
            return False
        
        return True
    
    def get_supported_timeframes(self) -> list[str]:
        """
        Get list of supported analysis timeframes
        
        Returns:
            List of supported timeframe strings
        """
        return ["1d", "1h", "4h", "1w"]  # Common timeframes supported by the system 