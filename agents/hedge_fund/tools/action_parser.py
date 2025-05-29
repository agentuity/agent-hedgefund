"""
Action Parser Module
Uses LLM with structured output to parse user queries into actionable items
"""

from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class ActionType(Enum):
    SEARCH_ASSET = "search_asset"
    ANALYZE_TRADE = "analyze_trade"
    GENERAL_QUESTION = "general_question"
    REFUSE_QUERY = "refuse_query"

class TradeIntent(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    ANALYZE = "analyze"

class ParsedAction(BaseModel):
    """Structured representation of user intent"""
    action_type: ActionType = Field(
        description="The type of action to take based on the user's query",
        examples=["search_asset", "analyze_trade", "general_question", "refuse_query"]
    )
    asset_query: Optional[str] = Field(
        default=None,
        description="The asset name or symbol to search for, if applicable",
        examples=["Tesla", "AAPL", "Bitcoin", "BTC-USD", None]
    )
    trade_intent: TradeIntent = Field(
        default=TradeIntent.ANALYZE,
        description="What the user wants to do with the asset",
        examples=["buy", "sell", "hold", "analyze"]
    )
    confidence: float = Field(
        description="Confidence level in the parsing decision",
        ge=0.0,
        le=1.0,
        examples=[0.95, 0.87, 0.76]
    )
    reasoning: str = Field(
        description="Explanation of why this action type was chosen",
        examples=[
            "User wants to buy Tesla stock, need to search for TSLA first",
            "User asking about general market concepts, no specific asset mentioned",
            "Weather question is not financial/trading related"
        ]
    )
    should_proceed: bool = Field(
        description="Whether the system should proceed with this action",
        examples=[True, False]
    )
    user_message: str = Field(
        description="The original user message for reference"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "action_type": "search_asset",
                    "asset_query": "Tesla",
                    "trade_intent": "buy",
                    "confidence": 0.95,
                    "reasoning": "User wants to buy Tesla stock, need to search for TSLA first",
                    "should_proceed": True,
                    "user_message": "Should I buy Tesla stock?"
                },
                {
                    "action_type": "general_question",
                    "asset_query": None,
                    "trade_intent": "analyze",
                    "confidence": 0.9,
                    "reasoning": "General educational question about markets",
                    "should_proceed": True,
                    "user_message": "How does the stock market work?"
                },
                {
                    "action_type": "refuse_query",
                    "asset_query": None,
                    "trade_intent": "analyze",
                    "confidence": 0.95,
                    "reasoning": "Weather question is not financial/trading related",
                    "should_proceed": False,
                    "user_message": "What's the weather like?"
                }
            ]
        }

class ActionParser:
    """Parses user queries into structured actions using LLM with structured output"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.structured_llm = self.llm.with_structured_output(
            ParsedAction,
            include_raw=True
        )
    
    def parse_query(self, user_message: str) -> ParsedAction:
        """
        Parse user query into structured action using LangChain's structured output
        """
        
        try:
            system_prompt = self._get_system_prompt()
            user_prompt = f"Parse this user message: '{user_message}'"
             
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            result = self.structured_llm.invoke(messages)
            
            if isinstance(result, dict) and 'parsed' in result:
                parsed_action = result['parsed']
                if hasattr(parsed_action, 'user_message'):
                    parsed_action.user_message = user_message
                return parsed_action
            else:
                if hasattr(result, 'user_message'):
                    result.user_message = user_message
                return result
                
        except Exception as e:
            print(f"Error in action parsing: {e}")
            return self._create_fallback_action(user_message, f"Parsing error: {e}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for action parsing"""
        
        return """You are a financial query action parser. Your job is to analyze user messages and determine what action should be taken.

AVAILABLE ACTIONS:
1. "search_asset" - User is asking about a specific asset (stock/crypto) that needs to be searched/validated
2. "analyze_trade" - User wants trade analysis after asset is confirmed to exist  
3. "general_question" - User is asking general financial questions not about specific assets
4. "refuse_query" - User is asking about non-financial topics or inappropriate requests

TRADE INTENTS:
- "buy" - User wants to know if they should buy
- "sell" - User wants to know if they should sell  
- "hold" - User wants to know if they should hold
- "analyze" - General analysis request

IMPORTANT RULES:
- If user mentions ANY specific company name, stock symbol, or crypto name → "search_asset"
- Only use "analyze_trade" if the asset is already confirmed to exist (this won't happen in first pass)
- "general_question" is for things like "how does the stock market work?" 
- "refuse_query" is for non-financial topics like weather, cooking, etc.

EXAMPLES:

User: "Should I buy Tesla stock?"
→ action_type: "search_asset", asset_query: "Tesla", trade_intent: "buy", confidence: 0.95, reasoning: "User wants to buy Tesla stock, need to search for TSLA first", should_proceed: true

User: "How is Bitcoin doing?"
→ action_type: "search_asset", asset_query: "Bitcoin", trade_intent: "analyze", confidence: 0.9, reasoning: "User wants analysis of Bitcoin, need to search for BTC first", should_proceed: true

User: "What's the weather like?"
→ action_type: "refuse_query", asset_query: null, trade_intent: "analyze", confidence: 0.95, reasoning: "Weather question is not financial/trading related", should_proceed: false

User: "How does the stock market work?"
→ action_type: "general_question", asset_query: null, trade_intent: "analyze", confidence: 0.9, reasoning: "General educational question about markets", should_proceed: true

User: "Tell me about RANDOMCOMPANY123"
→ action_type: "search_asset", asset_query: "RANDOMCOMPANY123", trade_intent: "analyze", confidence: 0.8, reasoning: "User asking about specific company, need to search even if likely invalid", should_proceed: true

Always provide the user_message field with the original message for reference."""

    def _create_fallback_action(self, user_message: str, error_msg: str) -> ParsedAction:
        """Create fallback action when parsing fails"""
        
        return ParsedAction(
            action_type=ActionType.REFUSE_QUERY,
            asset_query=None,
            trade_intent=TradeIntent.ANALYZE,
            confidence=0.0,
            reasoning=f"Error parsing query: {error_msg}",
            should_proceed=False,
            user_message=user_message
        )

# Global instance
action_parser = ActionParser()

def parse_user_query(user_message: str) -> ParsedAction:
    """Convenience function to parse user query"""
    return action_parser.parse_query(user_message) 