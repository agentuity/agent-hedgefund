"""
Enhanced Action Parser with LangChain Structured Output

Uses LangChain's with_structured_output() for direct LLM-to-Pydantic conversion.
Leverages LLM intelligence with rich contextual prompts instead of brittle regex parsing.
"""

from typing import Dict, Any, Optional
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.hedge_fund.models import ParsedAction, IntentType, TradeDirection

logger = logging.getLogger(__name__)

class ActionParser:
    """
    Enhanced action parser using LangChain structured output.
    Trusts LLM intelligence for comprehensive financial query understanding.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        # Create structured LLM that returns ParsedAction directly
        self.structured_llm = self.llm.with_structured_output(ParsedAction)
        logger.info(f"🤖 Initialized structured action parser with {model}")
    
    def parse_query(self, user_message: str, user_context: Optional[Dict[str, Any]] = None) -> ParsedAction:
        """
        Parse user query using LLM structured output.
        Returns ParsedAction directly - no fallbacks needed with proper prompting.
        
        Args:
            user_message: The user's financial query
            user_context: Optional context (portfolio, preferences, etc.)
        """
        
        logger.info(f"🔍 Parsing query with LLM structured output: {user_message}")
        
        # Create rich contextual prompt
        prompt = self._create_comprehensive_prompt(user_message, user_context)
        
        # Let LLM do the intelligent parsing
        parsed_action = self.structured_llm.invoke(prompt)
        
        logger.info(f"✅ LLM parsed intent: {parsed_action.intent_type.value}")
        logger.info(f"📊 Primary asset: {parsed_action.primary_asset}")
        logger.info(f"🎯 Confidence: {parsed_action.confidence:.1%}")
        
        return parsed_action
    
    def _create_comprehensive_prompt(self, user_message: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create rich, contextual prompt that gives the LLM maximum understanding.
        Explains all intent types, provides examples, and sets clear expectations.
        """
        
        context_section = ""
        if user_context:
            context_section = f"\n\nUSER CONTEXT:\n{self._format_user_context(user_context)}"
        
        return f"""You are an expert financial query parser. Analyze this user's query and extract comprehensive information into the specified JSON structure.

USER QUERY: "{user_message}"{context_section}

INTENT TYPES (choose the most appropriate):

1. **trade_analysis** - User wants trading advice/analysis for a specific asset
   - "Should I buy Tesla stock?"
   - "Is it a good time to sell my Apple shares?"
   - "What do you think about Bitcoin right now?"

2. **portfolio_review** - User asking about their overall portfolio
   - "How is my portfolio performing?"
   - "Is my portfolio too risky?"
   - "Should I rebalance my holdings?"

3. **risk_assessment** - User concerned about risk levels
   - "Is Tesla too volatile for me?"
   - "What are the risks of crypto investing?"
   - "How risky is this investment?"

4. **market_update** - User wants current market status/performance
   - "How is the market doing today?"
   - "What's happening with Tesla stock today?"
   - "Bitcoin price update"

5. **general_info** - Educational/informational questions
   - "What is RSI?"
   - "How does technical analysis work?"
   - "Explain options trading"

6. **invalid_query** - Non-financial topics
   - "How do I cook pasta?"
   - "What's the weather like?"

ASSET EXTRACTION GUIDELINES:
- Company names: "Tesla" → "Tesla", "Apple" → "Apple"  
- Stock symbols: "AAPL" → "AAPL", "TSLA" → "TSLA"
- Crypto: "Bitcoin" → "Bitcoin", "BTC-USD" → "BTC-USD"
- Be intelligent about context: "Tesla stock" → primary_asset: "Tesla"
- Ignore pronouns and common words: "I", "a", "the", "should", etc.

TRADE DIRECTION:
- **buy**: "should I buy", "thinking of purchasing", "want to invest in"
- **sell**: "should I sell", "time to exit", "thinking of selling"  
- **hold**: "should I hold", "keep my position"
- **unclear**: informational questions, market updates

CONFIDENCE SCORING (0.0-1.0):
- 0.9+: Very clear intent with specific asset ("Should I buy Tesla stock?")
- 0.7-0.9: Clear intent, some ambiguity ("What about Tesla?")
- 0.5-0.7: Moderate clarity ("How's the market?")
- 0.3-0.5: Unclear intent ("Tell me about investing")
- 0.0-0.3: Very vague or invalid queries

STRENGTH INDICATORS (0.0-1.0):
- **trade_intent_strength**: How strongly they express wanting to trade
  - 1.0: "Should I buy now?" | 0.5: "What do you think about..." | 0.0: "What is..."
- **portfolio_context_strength**: How much portfolio info they provide
  - 1.0: "I own 100 shares, should I buy more?" | 0.5: "My portfolio is..." | 0.0: No portfolio mention
- **risk_concern_level**: How much they express risk concerns  
  - 1.0: "Is this too risky for me?" | 0.5: "What are the risks?" | 0.0: No risk mention

Be intelligent and contextual. If uncertain about a field, make your best educated guess based on the context. The goal is comprehensive extraction that captures the user's true intent and all relevant information they've provided.

Extract everything you can detect, even if you're not 100% certain. Your analysis will guide important financial decisions."""

    def _format_user_context(self, user_context: Dict[str, Any]) -> str:
        """Format user context for the prompt"""
        context_parts = []
        
        if user_context.get('portfolio'):
            context_parts.append(f"Portfolio: {user_context['portfolio']}")
        if user_context.get('risk_tolerance'):
            context_parts.append(f"Risk Tolerance: {user_context['risk_tolerance']}")
        if user_context.get('preferences'):
            context_parts.append(f"Preferences: {user_context['preferences']}")
        if user_context.get('investment_timeline'):
            context_parts.append(f"Timeline: {user_context['investment_timeline']}")
            
        return "\n".join(context_parts) if context_parts else "No additional context provided"

# Global instance for backward compatibility
enhanced_action_parser = ActionParser()

def parse_user_query_enhanced(user_message: str, user_context: Optional[Dict[str, Any]] = None) -> ParsedAction:
    """Enhanced convenience function using structured output"""
    return enhanced_action_parser.parse_query(user_message, user_context)

def parse_user_query(user_message: str, user_context: Optional[Dict[str, Any]] = None) -> ParsedAction:
    """Main parsing function - clean interface with LLM intelligence"""
    return parse_user_query_enhanced(user_message, user_context) 