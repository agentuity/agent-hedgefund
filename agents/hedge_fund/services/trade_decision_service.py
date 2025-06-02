from typing import Dict, List, Optional, Any, TypedDict
from enum import Enum
import json
from datetime import datetime
import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from agents.hedge_fund.services.technical_analyst import (
    TechnicalAnalysisRequest, 
    TechnicalAnalysisToolOutput,
    run_technical_analysis_tool,
    AssetType,
    IndicatorSpec
)

from agents.hedge_fund.services.market_sentiment_analyst import (
    run_enhanced_market_sentiment_analysis,
    AssetType as SentimentAssetType
)

logger = logging.getLogger(__name__)

# --- Pydantic Schemas ---

class TradeDecision(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"
    NO_TRADE = "NO_TRADE"

class ConfidenceLevel(str, Enum):
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"        
    MEDIUM = "MEDIUM"
    LOW = "LOW"    
    VERY_LOW = "VERY_LOW"

class TradeRecommendation(BaseModel):
    symbol: str
    asset_type: AssetType
    decision: TradeDecision
    confidence: ConfidenceLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Detailed reasoning for the trade decision")
    key_factors: List[str] = Field(..., description="Key factors influencing the decision")
    market_context: str = Field(..., description="Current market context and conditions")
    entry_price_target: Optional[float] = None
    enhanced_insights: Optional[Dict[str, Any]] = None  # LLM insights from sentiment analysis
    timestamp: datetime = Field(default_factory=datetime.now)

class TradeAnalysisRequest(BaseModel):
    symbol: str
    asset_type: AssetType
    timeframe: str = "1d"
    current_portfolio_context: Optional[Dict[str, Any]] = None
    market_conditions: Optional[Dict[str, Any]] = None
    use_enhanced_sentiment: bool = Field(default=True, description="Enable LLM-powered sentiment analysis")
    parsed_action: Optional[Any] = Field(default=None, description="ParsedAction containing user risk tolerance and intent")

# --- LangGraph State ---

class TradeDecisionState(TypedDict):
    request: TradeAnalysisRequest
    technical_analysis: Optional[TechnicalAnalysisToolOutput]
    market_sentiment: Optional[Dict[str, Any]]
    enhanced_sentiment_insights: Optional[Dict[str, Any]]  # LLM insights
    options_flow: Optional[Dict[str, Any]]
    market_context: Optional[Dict[str, Any]]
    llm_analysis: Optional[str]
    final_recommendation: Optional[TradeRecommendation]
    error: Optional[str]

# --- Helper Functions ---

def create_technical_analysis_request(request: TradeAnalysisRequest) -> TechnicalAnalysisRequest:
    """Create technical analysis request with professional trading setup"""
    return TechnicalAnalysisRequest(
        asset_type=request.asset_type,
        symbol=request.symbol,
        timeframe=request.timeframe,
        indicators=[
            # EMA crossover system (8/21)
            IndicatorSpec(name="ema_crossover", params={"fast_period": 8, "slow_period": 21}),
            # Standard MACD (12,26,9)
            IndicatorSpec(name="macd", params={"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            # Custom MACD aligned with EMA setup (8,21,9)
            IndicatorSpec(name="macd", params={"fast_period": 8, "slow_period": 21, "signal_period": 9}),
            # RSI for overbought/oversold confirmation
            IndicatorSpec(name="rsi", params={"period": 14}),
            # Additional trend confirmation
            IndicatorSpec(name="sma", params={"period": 50}),
            IndicatorSpec(name="ema", params={"period": 200}),  # Long-term trend
            # Volume analysis
            IndicatorSpec(name="volume_ratio", params={"period": 20}),
            IndicatorSpec(name="volume_trend", params={"period": 5}),
        ]
    )

def calculate_signal_metrics(tech_analysis: TechnicalAnalysisToolOutput) -> Dict[str, Any]:
    """Calculate signal strength metrics from technical analysis"""
    signal_strengths = []
    trend_signals = []
    
    for indicator in tech_analysis.indicators:
        if indicator.signal:
            signal_strengths.append(indicator.signal.strength)
            if indicator.signal.signal.value in ["BUY", "SELL"]:
                trend_signals.append(indicator.signal.signal.value)
    
    avg_signal_strength = sum(signal_strengths) / len(signal_strengths) if signal_strengths else 0.0
    
    return {
        "signal_strengths": signal_strengths,
        "trend_signals": trend_signals,
        "avg_signal_strength": avg_signal_strength,
        "signal_count": len(signal_strengths)
    }

def determine_market_conviction(avg_signal_strength: float) -> str:
    """Determine market conviction level based on average signal strength"""
    if avg_signal_strength > 0.7:
        return "HIGH"
    elif avg_signal_strength > 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def determine_trend_direction(trend_signals: List[str]) -> str:
    """Determine overall trend direction from signals"""
    buy_count = trend_signals.count("BUY")
    sell_count = trend_signals.count("SELL")
    
    if buy_count > sell_count:
        return "BULLISH"
    elif sell_count > buy_count:
        return "BEARISH"
    else:
        return "NEUTRAL"

def analyze_signal_confluence(tech_analysis: TechnicalAnalysisToolOutput) -> Dict[str, Any]:
    """Analyze signal confluence and generate key factors"""
    buy_signals = 0
    sell_signals = 0
    total_strength = 0.0
    signal_count = 0
    key_factors = []
    
    for indicator in tech_analysis.indicators:
        if indicator.signal:
            signal_count += 1
            total_strength += indicator.signal.strength
            
            signal_type = indicator.signal.signal.value
            if signal_type == "BUY":
                buy_signals += 1
                key_factors.append(f"Bullish {indicator.name}: {indicator.signal.reason}")
            elif signal_type == "SELL":
                sell_signals += 1
                key_factors.append(f"Bearish {indicator.name}: {indicator.signal.reason}")
            elif signal_type == "HOLD":
                key_factors.append(f"Neutral {indicator.name}: {indicator.signal.reason}")
    
    # Calculate confluence score
    if signal_count > 0:
        avg_strength = total_strength / signal_count
        signal_agreement = max(buy_signals, sell_signals) / signal_count
        confluence_score = (avg_strength + signal_agreement) / 2
    else:
        confluence_score = 0.0
    
    return {
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "avg_strength": avg_strength,
        "confluence_score": confluence_score,
        "key_factors": key_factors[:5]  # Top 5 factors
    }

def determine_trade_decision(confluence_data: Dict[str, Any], parsed_action: Optional[Any] = None) -> tuple[TradeDecision, ConfidenceLevel]:
    """Determine trade decision and confidence based on confluence analysis and user risk tolerance from ParsedAction"""
    buy_signals = confluence_data["buy_signals"]
    sell_signals = confluence_data["sell_signals"]
    confluence_score = confluence_data["confluence_score"]
    
    # Dynamic thresholds based on ParsedAction risk indicators
    min_signals_required, min_confluence_score = _calculate_dynamic_thresholds(parsed_action)
    
    logger.info(f"🎯 Trade decision with dynamic thresholds:")
    logger.info(f"   📊 Signals: BUY={buy_signals}, SELL={sell_signals}, confluence={confluence_score:.2f}")
    logger.info(f"   🎚️ Thresholds: min_signals={min_signals_required}, min_confluence={min_confluence_score:.2f}")
    
    if buy_signals >= min_signals_required and confluence_score > min_confluence_score:
        high_conf_buffer = max(0.15, min_confluence_score * 0.4)
        if confluence_score > (min_confluence_score + high_conf_buffer):
            return TradeDecision.STRONG_BUY, ConfidenceLevel.HIGH
        else:
            return TradeDecision.BUY, ConfidenceLevel.MEDIUM
    elif sell_signals >= min_signals_required and confluence_score > min_confluence_score:
        high_conf_buffer = max(0.15, min_confluence_score * 0.4)
        if confluence_score > (min_confluence_score + high_conf_buffer):
            return TradeDecision.STRONG_SELL, ConfidenceLevel.HIGH
        else:
            return TradeDecision.SELL, ConfidenceLevel.MEDIUM
    elif abs(buy_signals - sell_signals) <= 1:
        if parsed_action and _is_risk_tolerant_user(parsed_action):
            if buy_signals > sell_signals:
                logger.info("🔥 Risk-tolerant user: choosing BUY on mixed signals")
                return TradeDecision.BUY, ConfidenceLevel.LOW
            elif sell_signals > buy_signals:
                logger.info("🔥 Risk-tolerant user: choosing SELL on mixed signals")
                return TradeDecision.SELL, ConfidenceLevel.LOW
        return TradeDecision.HOLD, ConfidenceLevel.LOW
    else:
        return TradeDecision.NO_TRADE, ConfidenceLevel.VERY_LOW

def _calculate_dynamic_thresholds(parsed_action: Optional[Any]) -> tuple[int, float]:
    """Calculate dynamic thresholds based on ParsedAction risk indicators"""
    
    min_signals = 1
    min_confluence = 0.25
    
    if not parsed_action:
        logger.info("📝 No ParsedAction - using more responsive default thresholds")
        return min_signals, min_confluence
    
    # Extract key risk indicators from ParsedAction
    trade_intent = getattr(parsed_action, 'trade_intent_strength', 0.0)
    risk_concern = getattr(parsed_action, 'risk_concern_level', 0.5)
    risk_tolerance = getattr(parsed_action, 'risk_tolerance_hint', None)
    urgency = getattr(parsed_action, 'urgency_level', 0.0)
    
    # Start with base adjustments based on numerical indicators
    # High trade intent = more aggressive
    intent_adjustment = trade_intent * 0.2
    
    # High risk concern = more conservative  
    concern_adjustment = risk_concern * 0.15
    
    # High urgency = slightly more aggressive
    urgency_adjustment = urgency * 0.1
    
    # Apply numerical adjustments
    confluence_adjustment = -intent_adjustment + concern_adjustment - urgency_adjustment
    min_confluence = max(0.1, min(0.75, min_confluence + confluence_adjustment))  # Lower minimum from 0.15 to 0.1
    
    # Signal count adjustments based on risk tolerance
    if risk_tolerance:
        risk_hint = risk_tolerance.lower()
        if any(word in risk_hint for word in ["aggressive", "risky", "high risk", 
            "willing to lose", "yolo", "gamble"]):
            logger.info("🔥 AGGRESSIVE trader detected!")
            min_signals = 1
            min_confluence = max(0.1, min_confluence - 0.25)
        elif any(word in risk_hint for word in ["moderate", "some risk", "medium risk"]):
            logger.info("📈 MODERATE risk trader detected")
            min_signals = 1
            min_confluence = max(0.15, min_confluence - 0.1)
        elif any(word in risk_hint for word in ["conservative", "safe", "low risk", "cautious"]):
            logger.info("🛡️ CONSERVATIVE trader detected")
            min_signals = 2
            min_confluence = min(0.5, min_confluence + 0.05)
    
    # Extreme combinations - make them actually work
    if trade_intent > 0.8 and risk_concern < 0.2:
        logger.info("🚀 HIGH intent + LOW concern = VERY AGGRESSIVE!")
        min_signals = 1
        min_confluence = max(0.05, min_confluence - 0.3)
    elif trade_intent < 0.3 and risk_concern > 0.7:
        logger.info("🐌 LOW intent + HIGH concern = VERY CONSERVATIVE!")
        min_signals = 2
        min_confluence = min(0.6, min_confluence + 0.15)
    
    min_signals = max(1, min(3, min_signals))
    min_confluence = max(0.05, min(0.8, min_confluence))
    
    logger.info(f"🎚️ Risk-adjusted thresholds calculated:")
    logger.info(f"   📊 Trade intent: {trade_intent:.2f}, Risk concern: {risk_concern:.2f}")
    logger.info(f"   🎯 Final: {min_signals} signals, {min_confluence:.2f} confluence")
    
    return min_signals, min_confluence

def _is_risk_tolerant_user(parsed_action: Any) -> bool:
    """Check if user is risk tolerant based on ParsedAction indicators"""
    
    trade_intent = getattr(parsed_action, 'trade_intent_strength', 0.0)
    risk_concern = getattr(parsed_action, 'risk_concern_level', 0.5)
    risk_tolerance = getattr(parsed_action, 'risk_tolerance_hint', None)
    
    # Explicit aggressive language
    if risk_tolerance:
        hint_lower = risk_tolerance.lower()
        if any(word in hint_lower for word in ["aggressive", "risky", "willing to lose", "yolo", "gamble"]):
            return True
    
    # High intent + low concern = risk tolerant
    if trade_intent > 0.6 and risk_concern < 0.4:
        return True
    
    return False

def create_llm_analysis_prompt(state: TradeDecisionState) -> str:
    """Create comprehensive analysis prompt for LLM including enhanced sentiment insights"""
    request = state["request"]
    tech_analysis = state["technical_analysis"]
    sentiment = state["market_sentiment"]
    enhanced_insights = state["enhanced_sentiment_insights"]
    options_flow = state["options_flow"]
    market_context = state["market_context"]
    
    # Build technical analysis section
    tech_section = []
    tech_section.append("TECHNICAL ANALYSIS:")
    
    # Get current price directly from technical analysis
    if tech_analysis and tech_analysis.current_price:
        tech_section.append(f"Current Price: ${tech_analysis.current_price:.2f}")
    else:
        tech_section.append("Current Price: N/A")
    
    tech_section.append("Technical Indicators:")
    
    if tech_analysis and tech_analysis.indicators:
        for indicator in tech_analysis.indicators:
            if indicator.signal:
                indicator_line = f"- {indicator.name}: {indicator.signal.signal.value} "
                indicator_line += f"(strength: {indicator.signal.strength:.2f}) - {indicator.signal.reason}"
                tech_section.append(indicator_line)
            elif indicator.error:
                tech_section.append(f"- {indicator.name}: ERROR - {indicator.error}")
            else:
                tech_section.append(f"- {indicator.name}: No signal available")
    else:
        tech_section.append("- No technical indicators available")
    
    # Build sentiment analysis section
    sentiment_section = []
    sentiment_section.append("MARKET SENTIMENT:")
    
    if sentiment and not sentiment.get("error"):
        overall_score = sentiment.get("overall_sentiment_score")
        overall_signal = sentiment.get("overall_sentiment_signal")
        
        if overall_signal:
            # Handle case where overall_signal is a string (sentiment value) rather than a dict
            if isinstance(overall_signal, str):
                sentiment_type = overall_signal
                confidence = sentiment.get("confidence", 0.0)
                sentiment_section.append(f"Overall: {sentiment_type} (confidence: {confidence:.2f})")
            else:
                # Handle case where overall_signal is a dict (shouldn't happen with current implementation)
                sentiment_type = overall_signal.get("sentiment", "UNKNOWN")
                confidence = overall_signal.get("confidence", 0.0)
                reason = overall_signal.get("reason", "No reason provided")
                sentiment_section.append(f"Overall: {sentiment_type} (confidence: {confidence:.2f}) - {reason}")
        else:
            sentiment_section.append(f"Overall Score: {overall_score:.3f}" if overall_score else "No overall sentiment")
        
        # Add individual source details
        sources = sentiment.get("sources", {})
        for source_name, source_data in sources.items():
            if source_data.get("available"):
                score = source_data.get("sentiment_score", 0.0)
                signal = source_data.get("sentiment_signal")
                data_points = source_data.get("data_points", 0)
                
                if signal:
                    # Handle case where signal is a string
                    if isinstance(signal, str):
                        signal_type = signal
                        signal_conf = source_data.get("confidence", 0.0)
                        sentiment_section.append(f"- {source_name}: {signal_type} (score: {score:.3f}, confidence: {signal_conf:.2f}, {data_points} data points)")
                    else:
                        # Handle case where signal is a dict
                        signal_type = signal.get("sentiment", "UNKNOWN")
                        signal_conf = signal.get("confidence", 0.0)
                        sentiment_section.append(f"- {source_name}: {signal_type} (score: {score:.3f}, confidence: {signal_conf:.2f}, {data_points} data points)")
                else:
                    sentiment_section.append(f"- {source_name}: Score {score:.3f} ({data_points} data points)")
            else:
                error = source_data.get("error", "Unknown error")
                sentiment_section.append(f"- {source_name}: ERROR - {error}")
    else:
        error_msg = sentiment.get("error", "No sentiment data available") if sentiment else "No sentiment data"
        sentiment_section.append(f"ERROR: {error_msg}")
    
    # Build enhanced insights section
    insights_section = []
    if enhanced_insights:
        insights_section.append("ENHANCED LLM SENTIMENT INSIGHTS:")
        
        if enhanced_insights.get("key_themes"):
            themes = enhanced_insights["key_themes"]
            insights_section.append(f"Key Themes: {', '.join(themes[:5])}")
        
        if enhanced_insights.get("market_moving_events"):
            events = enhanced_insights["market_moving_events"]
            insights_section.append(f"Market-Moving Events: {', '.join(events[:3])}")
        
        if enhanced_insights.get("risk_factors"):
            risks = enhanced_insights["risk_factors"]
            insights_section.append(f"Risk Factors: {', '.join(risks[:3])}")
        
        if enhanced_insights.get("opportunities"):
            opportunities = enhanced_insights["opportunities"]
            insights_section.append(f"Opportunities: {', '.join(opportunities[:3])}")
        
        llm_confidence = enhanced_insights.get("llm_confidence", 0.0)
        insights_section.append(f"LLM Analysis Confidence: {llm_confidence:.2f}")
    else:
        insights_section.append("ENHANCED INSIGHTS: No enhanced insights available")
    
    prompt_sections = [
        f"You are a professional hedge fund analyst making trading decisions. "
        f"Analyze the following data for {request.symbol} ({request.asset_type.value}):",
        "",
        "\n".join(tech_section),
        "",
        "\n".join(sentiment_section),
        "",
        "\n".join(insights_section),
        "",
        f"OPTIONS FLOW: {options_flow}",
        "",
        f"MARKET CONTEXT: {market_context}",
        "",
        "Based on this comprehensive analysis, provide:",
        "1. Your overall assessment of the trading opportunity",
        "2. Key confluence factors (where technical, sentiment, and LLM insights align)",
        "3. Main risks and concerns (integrate risk factors from enhanced insights)",
        "4. How sentiment supports or contradicts technical signals",
        "5. How the LLM-identified themes and events affect the trade thesis",
        "6. Your confidence level in any potential trade",
        "",
        "Be specific about WHY you would or wouldn't trade this setup. Focus on confluence of technical signals, sentiment analysis, and LLM-derived insights. Consider the market-moving events and risk factors identified by the enhanced analysis."
    ]
    
    prompt = "\n".join(prompt_sections)
    
    return prompt

def create_error_recommendation(request: TradeAnalysisRequest, error_msg: str) -> TradeRecommendation:
    """Create error recommendation when analysis fails"""
    return TradeRecommendation(
        symbol=request.symbol,
        asset_type=request.asset_type,
        decision=TradeDecision.NO_TRADE,
        confidence=ConfidenceLevel.VERY_LOW,
        confidence_score=0.0,
        reasoning=error_msg,
        key_factors=["System error"],
        market_context="System error"
    )

def convert_asset_type_for_sentiment(asset_type: AssetType) -> SentimentAssetType:
    """Convert AssetType to SentimentAssetType"""
    if asset_type == AssetType.CRYPTO:
        return SentimentAssetType.CRYPTO
    else:
        return SentimentAssetType.STOCK

# --- Node Functions ---

def gather_technical_analysis(state: TradeDecisionState) -> TradeDecisionState:
    """Gather technical analysis data using our technical analyst module"""
    try:
        request = state["request"]
        tech_request = create_technical_analysis_request(request)
        tech_result = run_technical_analysis_tool(tech_request)
        state["technical_analysis"] = tech_result
        logger.info(f"✅ Technical analysis completed for {request.symbol}")
        
    except Exception as e:
        state["error"] = f"Technical analysis failed: {str(e)}"
        logger.error(f"❌ Technical analysis error: {e}")
    
    return state

def gather_market_sentiment(state: TradeDecisionState) -> TradeDecisionState:
    """Gather comprehensive market sentiment analysis with enhanced LLM insights"""
    try:
        request = state["request"]
        
        # Convert asset type for sentiment analysis
        sentiment_asset_type = convert_asset_type_for_sentiment(request.asset_type)
        
        # Use enhanced sentiment analysis with LLM
        sentiment_result = run_enhanced_market_sentiment_analysis(
            symbol=request.symbol,
            asset_type=sentiment_asset_type,
            sources=["news_sentiment", "fear_greed_index", "economic_sentiment"],
            lookback_days=7,
            use_llm_insights=request.use_enhanced_sentiment
        )
        
        # Store sentiment data in expected format
        sentiment_data = {
            "overall_sentiment_score": sentiment_result.get("overall_sentiment_score"),
            "overall_sentiment_signal": sentiment_result.get("overall_sentiment_signal"),
            "confidence": sentiment_result.get("confidence", 0.0),
            "error": sentiment_result.get("error"),
            "sources": {}
        }
        
        # Process individual sources
        for source in sentiment_result.get("sources", []):
            if isinstance(source, dict):
                source_name = source.get("source", "unknown")
                sentiment_data["sources"][source_name] = {
                    "sentiment_signal": source.get("sentiment"),
                    "sentiment_score": source.get("score", 0.0),
                    "confidence": source.get("confidence", 0.0),
                    "data_points": source.get("data_points", 0),
                    "metadata": source.get("metadata", {}),
                    "available": True
                }
        
        state["market_sentiment"] = sentiment_data
        
        # Store enhanced LLM insights separately
        enhanced_insights = sentiment_result.get("enhanced_insights")
        if enhanced_insights:
            state["enhanced_sentiment_insights"] = enhanced_insights
            logger.info(f"🧠 Enhanced sentiment insights captured:")
            logger.info(f"  • Key themes: {enhanced_insights.get('key_themes', [])[:3]}")
            logger.info(f"  • Market events: {enhanced_insights.get('market_moving_events', [])[:2]}")
            logger.info(f"  • LLM confidence: {enhanced_insights.get('llm_confidence', 0.0):.2f}")
        else:
            state["enhanced_sentiment_insights"] = None
        
        # Log standard sentiment
        overall_signal = sentiment_result.get("overall_sentiment_signal")
        overall_confidence = sentiment_result.get("confidence", 0.0)
        if overall_signal:
            logger.info(f"📊 Market sentiment: {overall_signal} (confidence: {overall_confidence:.2f})")
        else:
            logger.info("📊 Market sentiment gathered (no overall signal)")
        
    except Exception as e:
        state["market_sentiment"] = {
            "error": f"Sentiment analysis failed: {str(e)}",
            "overall_sentiment_score": None,
            "overall_sentiment_signal": None,
            "sources": {}
        }
        state["enhanced_sentiment_insights"] = None
        logger.error(f"❌ Market sentiment error: {e}")
    
    return state

def gather_options_flow(state: TradeDecisionState) -> TradeDecisionState:
    """Placeholder for options flow analysis"""
    # TODO: Integrate with options flow data
    state["options_flow"] = {
        "gamma_exposure": "neutral",
        "put_call_ratio": 1.0,
        "max_pain": None,
        "flow_momentum": "neutral",
        "note": "Placeholder - to be implemented with real options data"
    }
    logger.info("📈 Options flow gathered (placeholder)")
    return state

def assess_market_context(state: TradeDecisionState) -> TradeDecisionState:
    """Assess overall market context and conditions"""
    try:
        tech_analysis = state["technical_analysis"]
        
        # Check if technical analysis is available and successful
        if not tech_analysis or tech_analysis.overall_signal.value == "UNCLEAR":
            state["market_context"] = {"error": "Cannot assess market context without valid technical data"}
            return state
        
        # Use the current_price field from technical analysis
        current_price = tech_analysis.current_price
        
        if not current_price:
            state["market_context"] = {"error": "No current price available in technical analysis"}
            return state
        
        signal_metrics = calculate_signal_metrics(tech_analysis)
        
        market_conviction = determine_market_conviction(signal_metrics["avg_signal_strength"])
        trend_direction = determine_trend_direction(signal_metrics["trend_signals"])
        
        state["market_context"] = {
            "market_conviction": market_conviction,
            "average_signal_strength": signal_metrics["avg_signal_strength"],
            "current_price": current_price,
            "signal_count": signal_metrics["signal_count"],
            "trend_direction": trend_direction
        }
        
        logger.info(f"📊 Market context assessed: {market_conviction} conviction, {trend_direction} trend")
        
    except Exception as e:
        state["market_context"] = {"error": f"Market context assessment failed: {str(e)}"}
        logger.error(f"❌ Market context error: {e}")
    
    return state

def analyze_trade_signal(state: TradeDecisionState) -> TradeDecisionState:
    """Use LLM to analyze all gathered data and provide reasoning"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        analysis_prompt = create_llm_analysis_prompt(state)
        
        messages = [HumanMessage(content=analysis_prompt)]
        response = llm.invoke(messages)
        
        state["llm_analysis"] = response.content
        logger.info("🤖 LLM analysis completed")
        logger.info(response.content)
        
    except Exception as e:
        state["llm_analysis"] = f"LLM analysis failed: {str(e)}"
        logger.error(f"❌ LLM analysis error: {e}")
    
    return state

def make_final_decision(state: TradeDecisionState) -> TradeDecisionState:
    """Make the final trade recommendation based on all analysis including enhanced insights"""
    try:
        request = state["request"]
        tech_analysis = state["technical_analysis"]
        market_context = state["market_context"]
        enhanced_insights = state["enhanced_sentiment_insights"]
        llm_analysis = state["llm_analysis"]
        
        # Check if technical analysis is available and successful
        if not tech_analysis or tech_analysis.overall_signal.value == "UNCLEAR":
            state["final_recommendation"] = create_error_recommendation(
                request, "Cannot make trade decision without valid technical data"
            )
            return state
        
        confluence_data = analyze_signal_confluence(tech_analysis)
        
        decision, confidence = determine_trade_decision(confluence_data, request.parsed_action)
        
        market_context_str = f"Market conviction: {market_context.get('market_conviction', 'UNKNOWN') if market_context else 'UNKNOWN'}. " \
                           f"Trend direction: {market_context.get('trend_direction', 'UNKNOWN') if market_context else 'UNKNOWN'}. " \
                           f"Signal count: {market_context.get('signal_count', 0) if market_context else 0}"
        
        # Enhanced reasoning including LLM insights
        base_reasoning = f"Signal confluence analysis: {confluence_data['buy_signals']} bullish, {confluence_data['sell_signals']} bearish signals. " \
                        f"Average signal strength: {confluence_data['avg_strength']:.2f}. "
        
        # Add enhanced insights to reasoning if available
        if enhanced_insights:
            insights_summary = []
            if enhanced_insights.get('key_themes'):
                insights_summary.append(f"Key market themes: {', '.join(enhanced_insights['key_themes'][:3])}")
            if enhanced_insights.get('market_moving_events'):
                insights_summary.append(f"Market events: {', '.join(enhanced_insights['market_moving_events'][:2])}")
            if enhanced_insights.get('risk_factors'):
                insights_summary.append(f"Risk factors: {', '.join(enhanced_insights['risk_factors'][:2])}")
            
            enhanced_reasoning = f"Enhanced insights: {'. '.join(insights_summary)}. "
            reasoning = base_reasoning + enhanced_reasoning + (llm_analysis or "")
        else:
            reasoning = base_reasoning + (llm_analysis or "")
        
        # Get current price from market context or use None as fallback
        entry_price = market_context.get('current_price') if market_context else None
        
        state["final_recommendation"] = TradeRecommendation(
            symbol=request.symbol,
            asset_type=request.asset_type,
            decision=decision,
            confidence=confidence,
            confidence_score=confluence_data["confluence_score"],
            reasoning=reasoning,
            key_factors=confluence_data["key_factors"],
            market_context=market_context_str,
            entry_price_target=entry_price,
            enhanced_insights=enhanced_insights  # Include full enhanced insights
        )
        
        logger.info(f"🎯 Final decision: {decision.value} with {confidence.value} confidence")
        if enhanced_insights:
            logger.info(f"🧠 Enhanced insights included in recommendation")
        
    except Exception as e:
        state["final_recommendation"] = create_error_recommendation(
            request, f"Decision making failed: {str(e)}"
        )
        state["error"] = f"Final decision failed: {str(e)}"
        logger.error(f"❌ Final decision error: {e}")
    
    return state

# --- LangGraph Workflow ---

def create_trade_decision_workflow() -> StateGraph:
    """Create the trade decision workflow using LangGraph"""
    workflow = StateGraph(TradeDecisionState)
    
    # Add nodes
    workflow.add_node("gather_technical", gather_technical_analysis)
    workflow.add_node("gather_sentiment", gather_market_sentiment)
    workflow.add_node("gather_options", gather_options_flow)
    workflow.add_node("assess_context", assess_market_context)
    workflow.add_node("analyze_trade_signal", analyze_trade_signal)
    workflow.add_node("final_decision", make_final_decision)
    
    # Define the workflow
    workflow.set_entry_point("gather_technical")
    
    # Sequential data gathering and analysis
    workflow.add_edge("gather_technical", "gather_sentiment")
    workflow.add_edge("gather_sentiment", "gather_options")
    workflow.add_edge("gather_options", "assess_context")
    workflow.add_edge("assess_context", "analyze_trade_signal")
    workflow.add_edge("analyze_trade_signal", "final_decision")
    workflow.add_edge("final_decision", END)
    
    return workflow.compile()

# --- Main Function ---

def run_trade_decision_analysis(request: TradeAnalysisRequest) -> TradeRecommendation:
    """Run the complete trade decision analysis workflow"""
    logger.info(f"🚀 Starting trade decision analysis for {request.symbol}")
    
    # Create workflow
    workflow = create_trade_decision_workflow()
    
    # Initialize state
    initial_state: TradeDecisionState = {
        "request": request,
        "technical_analysis": None,
        "market_sentiment": None,
        "enhanced_sentiment_insights": None,  # Include enhanced insights
        "options_flow": None,
        "market_context": None,
        "llm_analysis": None,
        "final_recommendation": None,
        "error": None
    }
    
    # Run workflow
    final_state = workflow.invoke(initial_state)
    
    if final_state.get("error"):
        logger.error(f"❌ Workflow error: {final_state['error']}")
    
    recommendation = final_state.get("final_recommendation")
    if recommendation:
        logger.info(f"✅ Trade decision completed: {recommendation.decision.value}")
        return recommendation
    else:
        return create_error_recommendation(request, "Workflow failed to generate recommendation")

if __name__ == "__main__":
    request = TradeAnalysisRequest(
        symbol="AAPL",
        asset_type=AssetType.STOCK,
        timeframe="1d"
    )
    
    logger.info("=== TRADE DECISION ANALYSIS ===")
    recommendation = run_trade_decision_analysis(request)
    logger.info("\n=== FINAL RECOMMENDATION ===")
    logger.info(recommendation.model_dump_json(indent=2))
    
    # Example crypto analysis
    crypto_request = TradeAnalysisRequest(
        symbol="BTC-USD",
        asset_type=AssetType.CRYPTO,
        timeframe="1d"
    )
    
    logger.info("\n\n=== CRYPTO TRADE DECISION ANALYSIS ===")
    crypto_recommendation = run_trade_decision_analysis(crypto_request)
    logger.info("\n=== CRYPTO FINAL RECOMMENDATION ===")
    logger.info(crypto_recommendation.model_dump_json(indent=2)) 