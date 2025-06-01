<div align="center">
    <img src="https://raw.githubusercontent.com/agentuity/cli/refs/heads/main/.github/Agentuity.png" alt="Agentuity" width="100"/> <br/>
    <strong>Build Agents, Not Infrastructure</strong> <br/>
<br />
</div>

# 🏦 AI Hedge Fund Agent

A sophisticated AI-powered hedge fund trading agent built with **Agentuity** and **LangGraph** orchestration, delivering professional-grade trading recommendations through comprehensive market analysis and intelligent routing.

## ✨ Key Features

- 🧠 **Enhanced LLM Query Parsing** - Single smart call extracts portfolio context, risk signals, and trade intent
- 📊 **Multi-Modal Analysis** - Technical indicators (RSI, MACD, EMA) + sentiment analysis + news
- 🎯 **Intelligent Routing** - LangGraph workflow adapts based on query complexity
- 💼 **Portfolio Awareness** - Understands existing positions and provides context-aware advice
- ⚡ **Real-Time Data** - Live market data from yfinance and CoinGecko APIs
- 🛡️ **Risk Detection** - Identifies risk concerns and routes to appropriate analysis
- 🎨 **Professional Formatting** - Beautiful, actionable responses with insights and recommendations
- 🚀 **Agentuity Native** - Built for seamless deployment on Agentuity platform

## 🏗️ Architecture Overview

```mermaid
graph TD
    A[User Query] --> B[agent.py<br/>Networking Layer]
    
    B --> C[controller.py<br/>LangGraph Orchestrator]
    
    C --> D[action_parser.py<br/>Enhanced LLM Parsing]
    D --> E{Query Analysis}
    
    E -->|Trade Analysis| F[asset_search_service.py<br/>Asset Search & Validation]
    E -->|General Info| K[response_formatter_service.py<br/>Response Formatting]
    E -->|Invalid Query| K
    
    F --> G[trade_decision_service.py<br/>Trade Analysis Hub]
    
    G --> H[technical_analyst.py<br/>RSI, MACD, EMA Analysis]
    G --> I[market_sentiment_analyst.py<br/>News & Sentiment Analysis]
    
    I --> I1[sentiment/news_analyzer.py<br/>NewsAPI Analysis]
    I --> I2[sentiment/fear_greed_analyzer.py<br/>CNN Fear/Greed Index]
    I --> I3[sentiment/economic_analyzer.py<br/>Economic Indicators]
    
    H --> J[Trade Recommendation]
    I --> J
    J --> K
    
    K --> L[Formatted Response]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style G fill:#fce4ec
    style K fill:#f1f8e9
```

## 📂 Project Structure

```
agents/hedge_fund/
├── agent.py                           # 🚀 Entry Point (Agentuity networking layer)
├── services/
│   ├── controller.py                  # 🎛️ LangGraph Orchestrator
│   ├── asset_search_service.py        # 🔍 Asset Search & Validation
│   ├── response_formatter_service.py  # 🎨 Response Formatting
│   ├── trade_decision_service.py      # 📊 Trade Analysis Hub
│   ├── technical_analyst.py           # 📈 Technical Indicators
│   └── market_sentiment_analyst.py    # 📰 Sentiment Analysis
├── tools/
│   ├── action_parser.py               # 🧠 Enhanced LLM Parsing
│   ├── asset_search.py                # 🔧 Asset Search Tools
│   └── sentiment/                     # 📊 Sentiment Analysis Tools
│       ├── base.py                    # Core sentiment classes
│       ├── manager.py                 # Sentiment orchestration
│       ├── news_analyzer.py           # NewsAPI integration
│       ├── fear_greed_analyzer.py     # CNN Fear/Greed index
│       └── economic_analyzer.py       # Economic indicators
├── models/                            # 🏗️ Data Models & Types
└── nodes/                             # 🔗 LangGraph Workflow Nodes
```

## 🚀 Quick Start with Agentuity

### Prerequisites

- **Python**: 3.10+
- **UV**: 0.5.25+ ([Documentation](https://docs.astral.sh/uv/))
- **Agentuity CLI**: Install from [agentuity.dev](https://agentuity.dev)

### Authentication

```bash
agentuity login
```

### Development Mode

Start the agent in development mode for real-time testing:

```bash
agentuity dev
```

This launches the Agentuity Console where you can test queries and see responses in real-time.

### Production Deployment

Deploy your agent to the Agentuity cloud platform:

```bash
agentuity deploy
```

### Example Queries

The agent handles various types of trading queries:

```
🔹 Simple Analysis: "Should I buy Apple stock?"
🔹 Portfolio Context: "I own 100 shares of Tesla, should I buy more?"
🔹 Risk Assessment: "Is my portfolio too risky with 50% tech stocks?"
🔹 Market Updates: "How is Bitcoin doing today?"
🔹 General Info: "What is RSI?"
🔹 Comparisons: "AAPL vs MSFT which is better?"
```

## 🧠 Enhanced Query Processing

Our **single smart LLM call** extracts comprehensive information:

```python
ParsedAction:
├── intent_type: "trade_analysis" | "portfolio_review" | "risk_assessment"
├── primary_asset: "Tesla" | "Bitcoin" | "AAPL"
├── mentioned_positions: [{"asset": "Tesla", "quantity": "100 shares"}]
├── risk_keywords: ["risky", "diversification", "volatile"]
├── quantities: [{"amount": "50", "unit": "percent"}]
├── trade_intent_strength: 0.9 (high conviction)
├── portfolio_context_strength: 0.8 (strong portfolio context)
└── risk_concern_level: 0.6 (moderate risk concern)
```

## 📊 Analysis Pipeline

### 1. Technical Analysis
- **SMA/EMA**: 8/21 crossover system
- **RSI**: Overbought/oversold signals (14-period)
- **MACD**: Trend confirmation and momentum
- **Volume Analysis**: Trend confirmation

### 2. Sentiment Analysis
- **News Sentiment**: Real-time news analysis via NewsAPI
- **Fear/Greed Index**: CNN market sentiment indicator
- **Economic Indicators**: VIX, bond yields, economic data

### 3. Enhanced Insights
- **AI-Generated Themes**: Market-moving narratives
- **Risk Factor Detection**: Potential downside risks
- **Event Awareness**: Earnings, Fed meetings, economic releases

## 🎯 Intelligent Routing

The LangGraph controller routes queries based on extracted information:

```mermaid
graph LR
    A[Parsed Query] --> B{Intent Type}
    
    B -->|TRADE_ANALYSIS| C[Asset Search → Trade Analysis]
    B -->|PORTFOLIO_REVIEW| D[Portfolio Manager*]
    B -->|RISK_ASSESSMENT| E[Risk Manager*]
    B -->|GENERAL_INFO| F[Direct Response]
    B -->|INVALID_QUERY| G[Rejection]
    
    C --> H[Enhanced Response]
    D --> H
    E --> H
    F --> H
    
    style D fill:#ffecb3
    style E fill:#ffecb3
```
*Future enhancements

## 🛡️ Risk Management (Future)

Planned risk management capabilities:
- Portfolio diversification analysis
- Position sizing recommendations
- Risk/reward ratio calculations
- Correlation analysis
- VaR (Value at Risk) calculations

## 🏗️ Deployment with Agentuity

### Local Development
```bash
# Start development server with hot reload
agentuity dev

# Test specific queries
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Should I buy Apple stock?"}'
```

### Production Deployment
```bash
# Deploy to Agentuity cloud
agentuity deploy

# View deployment status
agentuity status

# View logs
agentuity logs
```

### Environment Configuration
```bash
# Set API keys for production
agentuity env set NEWS_API_KEY your_news_api_key
agentuity env set OPENAI_API_KEY your_openai_key
```

## 📈 Example Output

```
🚀 Good timing for buying Apple Inc (AAPL)!

📈 Decision: BUY
🎯 Confidence: HIGH (87.5%)
💰 Current Price: $195.32
🏢 Exchange: NASDAQ

Key Factors:
1. Strong EMA 8/21 bullish crossover with increasing momentum
2. RSI at healthy 58.7 level with room for upside
3. Positive sentiment from strong earnings and iPhone demand

AI Insights:
🧠 Market Themes: AI integration, services growth, China recovery
📰 Key Events: Q4 earnings beat, new Vision Pro launch
⚠️ Risk Factors: Rising interest rates, China tensions

Analysis: Technical indicators show strong bullish momentum with the EMA crossover 
confirming the uptrend. Sentiment analysis reveals positive market reception to 
recent earnings and product launches...

*Analysis completed at 2024-01-15 14:30:22*
```

## 🤝 Contributing

This is a production-ready hedge fund agent with clean, modular architecture. Key principles:

- **Single Responsibility**: Each service has one clear purpose
- **Clean Dependencies**: No circular imports or legacy code
- **Enhanced Parsing**: Rich context extraction from user queries
- **Professional Output**: Actionable, well-formatted responses
- **Agentuity Native**: Built for optimal Agentuity platform performance

## 📚 Documentation

- [Agentuity Documentation](https://agentuity.dev/docs)
- [Agentuity Python SDK](https://agentuity.dev/SDKs/python)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)

## 🆘 Support

- [Agentuity Discord Community](https://discord.com/invite/vtn3hgUfuc)
- [Agentuity Support](https://agentuity.dev/support)

---

<div align="center">
<strong>Built with ❤️ using Agentuity, LangGraph, and advanced AI techniques</strong>
</div>
