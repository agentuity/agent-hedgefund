"""
User Context Models

Defines comprehensive user context including risk tolerance, portfolio information,
and trading preferences for personalized hedge fund analysis.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class RiskTolerance(str, Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

class InvestmentStyle(str, Enum):
    """Investment style preferences"""
    VALUE = "value"
    GROWTH = "growth"
    MOMENTUM = "momentum"
    DIVIDEND = "dividend"
    ESG = "esg"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"

class TimeHorizon(str, Enum):
    """Investment time horizons"""
    SHORT_TERM = "short_term"      # < 1 year
    MEDIUM_TERM = "medium_term"    # 1-5 years
    LONG_TERM = "long_term"        # > 5 years

class AssetClass(str, Enum):
    """Asset classes for portfolio allocation"""
    STOCKS = "stocks"
    CRYPTO = "crypto"
    BONDS = "bonds"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    CASH = "cash"

# --- Portfolio Models ---

class PortfolioPosition(BaseModel):
    """Individual position in the portfolio"""
    symbol: str
    asset_type: str  # "STOCK", "CRYPTO", etc.
    quantity: float
    average_cost: float
    current_value: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    percentage_of_portfolio: Optional[float] = None
    purchase_date: Optional[datetime] = None
    notes: Optional[str] = None

class PortfolioAllocation(BaseModel):
    """Portfolio allocation across asset classes"""
    stocks_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    crypto_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    bonds_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    cash_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    other_percentage: float = Field(default=0.0, ge=0.0, le=100.0)

class Portfolio(BaseModel):
    """Complete portfolio information"""
    total_value: float
    available_cash: float
    positions: List[PortfolioPosition] = Field(default_factory=list)
    allocation: PortfolioAllocation = Field(default_factory=PortfolioAllocation)
    last_updated: datetime = Field(default_factory=datetime.now)

    def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """Get a specific position by symbol"""
        for position in self.positions:
            if position.symbol.upper() == symbol.upper():
                return position
        return None

    def has_position(self, symbol: str) -> bool:
        """Check if portfolio contains a position"""
        return self.get_position(symbol) is not None

    def get_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure (placeholder - would need real sector data)"""
        # This would be implemented with real sector classification
        return {"technology": 40.0, "healthcare": 20.0, "financial": 15.0, "other": 25.0}

# --- Risk Profile Models ---

class RiskLimits(BaseModel):
    """Risk management limits"""
    max_position_size_percentage: float = Field(default=10.0, ge=0.0, le=100.0, description="Max % of portfolio in single position")
    max_sector_exposure_percentage: float = Field(default=30.0, ge=0.0, le=100.0, description="Max % exposure to single sector")
    max_single_trade_amount: float = Field(default=10000.0, ge=0.0, description="Max dollar amount for single trade")
    stop_loss_percentage: float = Field(default=10.0, ge=0.0, le=100.0, description="Default stop loss %")
    daily_loss_limit: float = Field(default=5000.0, ge=0.0, description="Max daily loss tolerance")

class TradingPreferences(BaseModel):
    """Trading style preferences"""
    preferred_timeframes: List[str] = Field(default=["1d"], description="Preferred analysis timeframes")
    investment_styles: List[InvestmentStyle] = Field(default=[InvestmentStyle.GROWTH], description="Preferred investment styles")
    avoid_penny_stocks: bool = Field(default=True, description="Avoid stocks under $5")
    avoid_high_volatility: bool = Field(default=False, description="Avoid high volatility assets")
    esg_preferences: bool = Field(default=False, description="Prefer ESG-compliant investments")
    dividend_preference: bool = Field(default=False, description="Prefer dividend-paying stocks")

# --- Complete User Context ---

class UserContext(BaseModel):
    """
    Complete user context for personalized analysis
    Contains risk tolerance, portfolio, and preferences
    """
    # Identity
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    
    # Risk Profile
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    risk_limits: RiskLimits = Field(default_factory=RiskLimits)
    
    # Portfolio
    portfolio: Optional[Portfolio] = None
    
    # Preferences
    trading_preferences: TradingPreferences = Field(default_factory=TradingPreferences)
    
    # Additional Context
    investment_goals: List[str] = Field(default_factory=list, description="Investment objectives")
    excluded_symbols: List[str] = Field(default_factory=list, description="Symbols to avoid")
    watchlist: List[str] = Field(default_factory=list, description="Symbols to monitor")
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)

    def get_available_cash_for_investment(self) -> float:
        """Get available cash considering risk limits"""
        if not self.portfolio:
            return 0.0
        
        # Consider keeping some cash reserve based on risk tolerance
        cash_reserve_percentage = {
            RiskTolerance.CONSERVATIVE: 0.3,
            RiskTolerance.MODERATE: 0.2,
            RiskTolerance.AGGRESSIVE: 0.1,
            RiskTolerance.VERY_AGGRESSIVE: 0.05
        }
        
        reserve_factor = cash_reserve_percentage.get(self.risk_tolerance, 0.2)
        reserve_amount = self.portfolio.available_cash * reserve_factor
        
        return max(0.0, self.portfolio.available_cash - reserve_amount)

    def can_invest_amount(self, amount: float) -> bool:
        """Check if user can invest the specified amount"""
        available = self.get_available_cash_for_investment()
        return amount <= min(available, self.risk_limits.max_single_trade_amount)

    def get_max_position_size(self) -> float:
        """Get maximum position size based on portfolio value and risk limits"""
        if not self.portfolio:
            return self.risk_limits.max_single_trade_amount
        
        max_by_percentage = self.portfolio.total_value * (self.risk_limits.max_position_size_percentage / 100)
        return min(max_by_percentage, self.risk_limits.max_single_trade_amount)

    def should_avoid_asset(self, symbol: str, price: Optional[float] = None) -> tuple[bool, str]:
        """Check if asset should be avoided based on user preferences"""
        # Check excluded symbols
        if symbol.upper() in [s.upper() for s in self.excluded_symbols]:
            return True, f"{symbol} is in your excluded symbols list"
        
        # Check penny stock preference
        if self.trading_preferences.avoid_penny_stocks and price and price < 5.0:
            return True, f"{symbol} is a penny stock (${price:.2f}) and you prefer to avoid them"
        
        return False, ""

# --- Helper Functions ---

def create_sample_user_context() -> UserContext:
    """Create a sample user context for testing"""
    return UserContext(
        user_id="sample_user",
        user_name="Sample User",
        risk_tolerance=RiskTolerance.MODERATE,
        time_horizon=TimeHorizon.MEDIUM_TERM,
        portfolio=Portfolio(
            total_value=100000.0,
            available_cash=10000.0,
            positions=[
                PortfolioPosition(
                    symbol="AAPL",
                    asset_type="STOCK",
                    quantity=100,
                    average_cost=150.0,
                    current_value=160.0,
                    market_value=16000.0,
                    percentage_of_portfolio=16.0
                ),
                PortfolioPosition(
                    symbol="BTC-USD",
                    asset_type="CRYPTO",
                    quantity=1.5,
                    average_cost=30000.0,
                    current_value=35000.0,
                    market_value=52500.0,
                    percentage_of_portfolio=52.5
                )
            ],
            allocation=PortfolioAllocation(
                stocks_percentage=45.0,
                crypto_percentage=40.0,
                cash_percentage=15.0
            )
        ),
        investment_goals=[
            "Long-term wealth building",
            "Retirement planning",
            "Diversified growth"
        ],
        watchlist=["MSFT", "TSLA", "ETH-USD"]
    )

def parse_user_context_from_dict(data: Dict[str, Any]) -> UserContext:
    """Parse user context from dictionary (useful for API requests)"""
    try:
        return UserContext.model_validate(data)
    except Exception as e:
        # Return default context if parsing fails
        return UserContext() 