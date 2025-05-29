"""
Constants for the Hedge Fund Agent
Contains all static messages, emojis, and configuration values
"""

from typing import Dict

# Response templates for general questions
GENERAL_RESPONSES: Dict[str, str] = {
    "market": "The stock market is a platform where shares of publicly traded companies are bought and sold. I can help you analyze specific stocks or crypto assets for trading decisions.",
    "trading": "Trading involves buying and selling financial assets to profit from price movements. I can analyze specific assets and provide trade recommendations based on technical and sentiment analysis.",
    "investing": "Investing is the practice of putting money into assets with the expectation of generating returns over time. I can help you analyze specific stocks or crypto for investment decisions.",
    "crypto": "Cryptocurrency is digital currency secured by cryptography. I can analyze specific crypto assets like Bitcoin, Ethereum, etc. for trading decisions.",
    "stocks": "Stocks represent ownership shares in companies. I can analyze specific stocks and provide buy/sell/hold recommendations based on technical and market sentiment analysis."
}

# Decision emojis mapping
DECISION_EMOJIS: Dict[str, str] = {
    "STRONG_BUY": "🚀",
    "BUY": "📈", 
    "HOLD": "⏸️",
    "SELL": "📉",
    "STRONG_SELL": "🔻",
    "NO_TRADE": "⛔"
}

# Status emojis
STATUS_EMOJIS = {
    "SUCCESS": "✅",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "INFO": "📚",
    "ANALYSIS": "📊"
}

# Default suggestion message
DEFAULT_SUGGESTION_MESSAGE = "Try asking about a specific asset like 'Should I buy Elon Musk stock?' or 'How is Bitcoin looking?'"

# Asset not found suggestions
ASSET_NOT_FOUND_SUGGESTIONS = [
    "Try a different spelling or the stock symbol (e.g., 'AAPL' for Apple)",
    "Make sure it's a publicly traded company or major cryptocurrency", 
    "Ask about popular assets like Tesla, Apple, Bitcoin, Ethereum, etc."
]

# Error messages
ERROR_MESSAGES = {
    "EMPTY_QUERY": "Please provide a message asking about a stock or crypto asset. For example: 'Should I buy Tesla?' or 'How is Bitcoin looking?'",
    "NO_ASSET_SPECIFIED": "I need to know which asset you're asking about. Please specify a stock or crypto asset.",
    "ANALYSIS_ERROR": "Sorry, I encountered an error analyzing {asset_name}. The asset exists but I couldn't complete the analysis. Please try again later.",
    "UNEXPECTED_ERROR": "Sorry, I encountered an unexpected error. Please try again later.",
    "NON_FINANCIAL_QUERY": "I'm a hedge fund trading assistant focused on stock and crypto analysis. I can't help with non-financial topics. Try asking about a specific asset like 'Should I buy Apple stock?' or 'How is Bitcoin doing?'"
}

# Default responses
DEFAULT_RESPONSES = {
    "GENERAL_HELP": "I'm a hedge fund trading assistant. I can help you analyze stocks and crypto assets for trading decisions. Try asking about a specific asset like 'Should I buy Tesla?' or 'How is Ethereum doing?'",
    "SEARCH_FIRST": "I need to search for the asset first. Please specify which stock or crypto you'd like me to analyze.",
    "UNCLEAR_REQUEST": "I'm not sure how to help with that. Please ask about a specific stock or crypto asset for trading analysis."
} 