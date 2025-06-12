<div align="center">
    <img src="https://raw.githubusercontent.com/agentuity/cli/refs/heads/main/.github/Agentuity.png" alt="Agentuity" width="100"/> <br/>
    <strong>Build Agents, Not Infrastructure</strong> <br/>
	<br/>
		<a target="_blank" href="https://app.agentuity.com/deploy" alt="Agentuity">
			<img src="https://app.agentuity.com/img/deploy.svg" /> 
		</a>

<br />
</div>
# 🏦 AI Hedge Fund Agent

A sophisticated AI-powered hedge fund trading agent built with **Agentuity** and **LangGraph** orchestration, delivering professional-grade trading decisions through comprehensive market analysis and intelligent portfolio management.

## ✨ Key Features

- 🧠 **Enhanced LLM Query Parsing** - Single smart call extracts portfolio context, risk signals, and trade intent
- 📊 **Multi-Modal Analysis** - Technical indicators (RSI, MACD, EMA) + sentiment analysis + real-time news
- 🎯 **Intelligent Routing** - LangGraph workflow adapts based on query complexity and user intent
- 💼 **Portfolio Decision Engine** - Makes concrete BUY/SELL/HOLD decisions with specific quantities
- ⚡ **Real-Time Data** - Live market data from yfinance and CoinGecko APIs
- 🛡️ **Risk Management** - Comprehensive risk assessment with position sizing recommendations
- 📋 **Actionable Output** - Concrete trading tables with entry/exit prices, stop losses, and position sizes
- 🚀 **Agentuity Native** - Built for seamless deployment on Agentuity platform

## 🔄 How It Works

The agent uses a 6-step LangGraph workflow: query parsing → asset search → technical analysis → risk assessment → portfolio decisions → response generation. It combines technical indicators (RSI, MACD, EMA) with sentiment analysis to make actionable trading recommendations.

📖 **[View Detailed Architecture](docs/architecture.md)**

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

Set up your `.env` with these keys:

```text
NEWSAPI_KEY=[Make one here: https://newsapi.org/]
FRED_API_KEY=[Make one here: https://fred.stlouisfed.org/docs/api/api_key.html]
AGENTUITY_SDK_KEY=
AGENTUITY_PROJECT_KEY=
```

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

The agent handles various types of trading queries with intelligent routing:

```
🔹 Trade Decisions: "Should I buy Apple stock?" → Full analysis + BUY/SELL table
🔹 Portfolio Context: "I own 100 shares of Tesla, should I buy more?" → Risk assessment
🔹 Market Analysis: "How is Bitcoin doing today?" → Technical + sentiment analysis  
🔹 Risk Assessment: "Is my portfolio too risky?" → Portfolio risk evaluation
```

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
agentuity env set NEWSAPI_KEY your_news_api_key # https://newsapi.org/
agentuity env set FRED_API_KEY your_openai_key # https://fred.stlouisfed.org/docs/api/api_key.html
```

## 🤝 Contributing

This is a production-ready hedge fund agent built with Agentuity and LangGraph. For detailed architecture and implementation principles, see the [architecture documentation](docs/architecture.md).

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
