"""
Query service for the Hedge Fund Agent
Handles user query parsing and routing logic
"""

from agents.hedge_fund.tools import parse_user_query, ParsedAction, ActionType


class QueryService:
    """
    Service responsible for parsing and categorizing user queries
    Encapsulates query processing logic
    """
    
    def parse_query(self, user_query: str) -> ParsedAction:
        """
        Parse user query using LLM and return structured action
        
        Args:
            user_query: Raw user input string
            
        Returns:
            ParsedAction with categorized intent and extracted information
        """
        if not user_query or not user_query.strip():
            # Return a mock ParsedAction for empty queries
            # This will be handled by the validation in the main agent
            raise ValueError("Empty query provided")
        
        return parse_user_query(user_query.strip())
    
    def requires_asset_search(self, action: ParsedAction) -> bool:
        """
        Determine if the parsed action requires asset search
        
        Args:
            action: Parsed user action
            
        Returns:
            True if asset search is needed, False otherwise
        """
        return action.action_type == ActionType.SEARCH_ASSET and action.asset_query
    
    def is_general_question(self, action: ParsedAction) -> bool:
        """
        Check if the action is a general financial question
        
        Args:
            action: Parsed user action
            
        Returns:
            True if it's a general question, False otherwise
        """
        return action.action_type == ActionType.GENERAL_QUESTION
    
    def should_refuse_query(self, action: ParsedAction) -> bool:
        """
        Check if the query should be refused (non-financial)
        
        Args:
            action: Parsed user action
            
        Returns:
            True if query should be refused, False otherwise
        """
        return action.action_type == ActionType.REFUSE_QUERY 