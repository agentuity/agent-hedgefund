"""
Formatters for the Hedge Fund Agent
Handles all presentation logic and complex formatting
"""

from agents.hedge_fund.agents.trade_decision_agent import TradeRecommendation
from agents.hedge_fund.tools import AssetInfo, TradeIntent
from agents.hedge_fund.utils.constants import DECISION_EMOJIS, STATUS_EMOJIS


class TradeRecommendationFormatter:
    """
    Formats trade recommendations into user-friendly responses
    Separates presentation logic from business logic
    """
    
    def format(self, recommendation: TradeRecommendation, asset_info: AssetInfo, trade_intent: TradeIntent) -> str:
        """
        Format a trade recommendation into a comprehensive user response
        
        Args:
            recommendation: Trade analysis result
            asset_info: Information about the analyzed asset
            trade_intent: User's intended trade action
            
        Returns:
            Formatted response string with all relevant information
        """
        symbol = recommendation.symbol
        decision = recommendation.decision.value
        confidence = recommendation.confidence.value
        confidence_score = recommendation.confidence_score
        
        # Build the response components
        intro = self._build_intro(asset_info, decision, trade_intent)
        decision_section = self._build_decision_section(decision, confidence, confidence_score)
        asset_info_section = self._build_asset_info_section(asset_info)
        factors_section = self._build_factors_section(recommendation.key_factors)
        analysis_section = self._build_analysis_section(recommendation)
        timestamp_section = self._build_timestamp_section(recommendation)
        
        # Combine all sections
        response_parts = [
            intro,
            "",
            decision_section,
            asset_info_section,
            "",
            factors_section,
            "",
            analysis_section,
            "",
            timestamp_section
        ]
        
        return "\n".join(filter(None, response_parts))
    
    def _build_intro(self, asset_info: AssetInfo, decision: str, trade_intent: TradeIntent) -> str:
        """Build the introductory section based on trade intent and decision"""
        symbol = asset_info.symbol
        name = asset_info.name
        
        if trade_intent == TradeIntent.BUY:
            if decision in ["STRONG_BUY", "BUY"]:
                return f"{STATUS_EMOJIS['SUCCESS']} **Good timing for buying {name} ({symbol})!**"
            elif decision == "HOLD":
                return f"{STATUS_EMOJIS['WARNING']} **Mixed signals for {name} - consider waiting**"
            else:
                return f"{STATUS_EMOJIS['ERROR']} **Not recommended to buy {name} right now**"
        elif trade_intent == TradeIntent.SELL:
            if decision in ["STRONG_SELL", "SELL"]:
                return f"{STATUS_EMOJIS['SUCCESS']} **Good timing to sell {name}**"
            elif decision == "HOLD":
                return f"{STATUS_EMOJIS['WARNING']} **Mixed signals for {name} - consider holding**"
            else:
                return f"{STATUS_EMOJIS['ERROR']} **Not recommended to sell {name} right now**"
        else:
            return f"{STATUS_EMOJIS['ANALYSIS']} **Analysis for {name} ({symbol})**"
    
    def _build_decision_section(self, decision: str, confidence: str, confidence_score: float) -> str:
        """Build the decision and confidence section"""
        decision_emoji = DECISION_EMOJIS.get(decision, "❓")
        return f"{decision_emoji} **Decision:** {decision}\n🎯 **Confidence:** {confidence} ({confidence_score:.1%})"
    
    def _build_asset_info_section(self, asset_info: AssetInfo) -> str:
        """Build the asset information section"""
        info_parts = []
        
        if asset_info.current_price:
            info_parts.append(f"💰 **Current Price:** ${asset_info.current_price:.2f}")
        
        if asset_info.exchange:
            info_parts.append(f"🏢 **Exchange:** {asset_info.exchange}")
        
        return "\n".join(info_parts)
    
    def _build_factors_section(self, key_factors: list) -> str:
        """Build the key factors section"""
        factors_text = "**Key Factors:**"
        
        for i, factor in enumerate(key_factors[:3], 1):  # Limit to top 3 factors
            factors_text += f"\n{i}. {factor}"
        
        return factors_text
    
    def _build_analysis_section(self, recommendation: TradeRecommendation) -> str:
        """Build the market context and analysis section"""
        return (
            f"**Market Context:** {recommendation.market_context}\n\n"
            f"**Analysis:** {recommendation.reasoning}"
        )
    
    def _build_timestamp_section(self, recommendation: TradeRecommendation) -> str:
        """Build the timestamp section"""
        return f"*Analysis completed at {recommendation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*" 