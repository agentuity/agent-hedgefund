"""
Response builder for the Hedge Fund Agent
Centralizes all response generation logic with clear separation of concerns
"""

from typing import List
from agents.hedge_fund.tools import ParsedAction
from agents.hedge_fund.utils.constants import (
    GENERAL_RESPONSES, ASSET_NOT_FOUND_SUGGESTIONS, 
    DEFAULT_SUGGESTION_MESSAGE, ERROR_MESSAGES, DEFAULT_RESPONSES,
    STATUS_EMOJIS
)


class ResponseBuilder:
    """
    Builds different types of responses for the hedge fund agent
    Follows the Builder pattern for clean response construction
    """
    
    def build_general_response(self, action: ParsedAction) -> str:
        """
        Build response for general financial questions
        
        Args:
            action: Parsed user action containing the query
            
        Returns:
            Formatted response with suggestions
        """
        query_lower = action.user_message.lower()
        
        for keyword, response in GENERAL_RESPONSES.items():
            if keyword in query_lower:
                return f"{STATUS_EMOJIS['INFO']} **{response}**\n\n{DEFAULT_SUGGESTION_MESSAGE}"
        
        return f"{DEFAULT_RESPONSES['GENERAL_HELP']}\n\n{DEFAULT_SUGGESTION_MESSAGE}"
    
    def build_asset_not_found_response(self, action: ParsedAction) -> str:
        """
        Build response when asset search fails
        
        Args:
            action: Parsed user action containing the asset query
            
        Returns:
            Formatted error response with suggestions
        """
        suggestions_text = self._format_suggestions_list(ASSET_NOT_FOUND_SUGGESTIONS)
        
        return (
            f"{STATUS_EMOJIS['ERROR']} **Asset Not Found**\n\n"
            f"I couldn't find '{action.asset_query}' in the market data.\n\n"
            f"**Suggestions:**\n{suggestions_text}\n\n"
            f"*I can analyze major stocks and cryptocurrencies that are actively traded.*"
        )
    
    def build_error_response(self, error_type: str, **kwargs) -> str:
        """
        Build standardized error responses
        
        Args:
            error_type: Type of error from ERROR_MESSAGES
            **kwargs: Format parameters for the error message
            
        Returns:
            Formatted error response
        """
        if error_type in ERROR_MESSAGES:
            message = ERROR_MESSAGES[error_type].format(**kwargs)
            return f"{STATUS_EMOJIS['ERROR']} {message}"
        
        return f"{STATUS_EMOJIS['ERROR']} {ERROR_MESSAGES['UNEXPECTED_ERROR']}"
    
    def build_simple_response(self, response_type: str) -> str:
        """
        Build simple responses from DEFAULT_RESPONSES
        
        Args:
            response_type: Key from DEFAULT_RESPONSES
            
        Returns:
            Formatted response
        """
        if response_type in DEFAULT_RESPONSES:
            return DEFAULT_RESPONSES[response_type]
        
        return DEFAULT_RESPONSES['UNCLEAR_REQUEST']
    
    def _format_suggestions_list(self, suggestions: List[str]) -> str:
        """
        Format a list of suggestions with bullet points
        
        Args:
            suggestions: List of suggestion strings
            
        Returns:
            Formatted suggestions string
        """
        return "\n".join([f"• {suggestion}" for suggestion in suggestions]) 