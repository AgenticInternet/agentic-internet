# Agentic Internet - Features Documentation

## 🚀 Overview

Agentic Internet is an advanced AI agent framework that provides multiple agent types, specialized agents for different use cases, and comprehensive tool integration including BrowserUse for web automation.

## 🎯 Agent Types

### 1. ToolCallingAgent
- **Best for**: Tool-heavy tasks, web searches, API interactions
- **Features**: Efficient tool use, structured responses
- **Use when**: You need to interact with multiple tools and APIs

```python
from agentic_internet import InternetAgent

agent = InternetAgent(
    agent_type="tool_calling",
    model_id="openrouter/anthropic/claude-sonnet-4"
)
```

### 2. CodeAgent
- **Best for**: Data analysis, code generation, mathematical computations
- **Features**: Can execute Python code, manipulate data, create visualizations
- **Use when**: You need to process data or generate/execute code

```python
agent = InternetAgent(
    agent_type="code",
    model_id="openrouter/anthropic/claude-sonnet-4",
    additional_authorized_imports=["scipy", "sklearn"]  # Add extra imports
)
```

## 🛠️ Specialized Agents

### BrowserAutomationAgent
Web automation specialist using Browser Use Cloud API.

```python
from agentic_internet import BrowserAutomationAgent

agent = BrowserAutomationAgent()

# Scrape structured data
data = agent.scrape_structured_data(
    url="https://news.ycombinator.com",
    data_schema={
        "posts": [{"title": "string", "url": "string", "points": "number"}]
    }
)

# Fill and submit forms
result = agent.fill_form(
    url="https://example.com/form",
    form_data={"name": "John Doe", "email": "john@example.com"}
)

# Monitor websites for changes
monitoring = agent.monitor_website(
    url="https://example.com",
    check_for="price changes",
    interval_minutes=30
)
```

### DataAnalysisAgent
Specialized for data analysis and visualization tasks.

```python
from agentic_internet import DataAnalysisAgent

agent = DataAnalysisAgent()

# Analyze datasets
analysis = agent.analyze_dataset(
    data=[
        {"month": "Jan", "sales": 45000, "costs": 32000},
        {"month": "Feb", "sales": 52000, "costs": 35000}
    ],
    analysis_type="comprehensive"
)

# Compare datasets
comparison = agent.compare_datasets(
    dataset1=data1,
    dataset2=data2,
    comparison_criteria=["trends", "averages", "outliers"]
)
```

### ContentCreationAgent
Optimized for writing and content generation.

```python
from agentic_internet import ContentCreationAgent

agent = ContentCreationAgent()

# Write articles
article = agent.write_article(
    topic="The Future of AI",
    style="informative",
    word_count=500,
    sources_required=True
)

# Summarize content
summary = agent.summarize_content(
    content=article["content"],
    summary_type="bullet_points"
)
```

### MarketResearchAgent
Market research and competitive analysis specialist.

```python
from agentic_internet import MarketResearchAgent

agent = MarketResearchAgent()

# Analyze competitors
analysis = agent.analyze_competitor(
    company_name="OpenAI",
    aspects=["products", "pricing", "market_position"]
)

# Analyze market trends
trends = agent.market_trends(
    industry="AI and Machine Learning",
    timeframe="2024"
)
```

### TechnicalSupportAgent
Technical support and troubleshooting specialist.

```python
from agentic_internet import TechnicalSupportAgent

agent = TechnicalSupportAgent()

# Troubleshoot problems
solution = agent.troubleshoot(
    problem_description="Python script throws ImportError",
    system_info={"OS": "MacOS", "Python": "3.11"}
)

# Review code
review = agent.code_review(
    code="def add(a, b): return a + b",
    language="python",
    focus_areas=["performance", "error_handling"]
)
```

## 🔧 Tools

### Web Tools
- **WebSearchTool**: DuckDuckGo web search
- **WebScraperTool**: Extract content from websites
- **NewsSearchTool**: Search for news articles

### Browser Use Tools (with API key)
- **BrowserUseTool**: Basic browser automation
- **AsyncBrowserUseTool**: Async browser operations with streaming
- **StructuredBrowserUseTool**: Extract structured data from websites

### Code Execution Tools
- **PythonExecutorTool**: Execute Python code safely
- **DataAnalysisTool**: Analyze data with pandas

### SerpAPI Tools (with API key)
- **GoogleSearchTool**: Google search with filters
- **GoogleShoppingTool**: Product search
- **GoogleMapsLocalTool**: Local business search
- **GoogleScholarTool**: Academic paper search
- **MultiEngineSearchTool**: Search across multiple engines

## 🔑 API Keys Configuration

Create a `.env` file in your project root:

```env
# Model Provider Keys
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_TOKEN=hf_...

# Tool API Keys
BROWSER_USE_API_KEY=bu_...
SERPAPI_API_KEY=...
```

## 📦 Installation

```bash
# Install base package
uv add agentic-internet

# Optional: Install visualization libraries for DataAnalysisAgent
uv add matplotlib seaborn plotly

# Optional: Install Browser Use SDK
uv add browser-use-sdk
```

## 🎮 Usage Examples

### Basic Usage
```python
from agentic_internet import InternetAgent

# Create an agent
agent = InternetAgent(
    model_id="openrouter/anthropic/claude-sonnet-4",
    agent_type="tool_calling",
    verbose=True
)

# Run a task
result = agent.run("Search for the latest AI news and summarize it")
```

### Advanced Workflow
```python
from agentic_internet import (
    MarketResearchAgent,
    ContentCreationAgent
)

# Research phase
researcher = MarketResearchAgent()
trends = researcher.market_trends("AI", "2024")

# Content creation phase
writer = ContentCreationAgent()
article = writer.write_article(
    topic=f"AI Market Trends: {trends['trends'][:100]}...",
    style="technical",
    word_count=1000
)
```

### Interactive Chat
```python
from agentic_internet import InternetAgent

agent = InternetAgent()
agent.chat()  # Start interactive session
```

## 🏗️ Architecture

```
agentic_internet/
├── agents/
│   ├── internet_agent.py      # Base agent with configurable types
│   ├── specialized_agents.py  # Specialized agent implementations
│   └── multi_model_serpapi.py # Multi-model orchestration
├── tools/
│   ├── web_search.py          # Web search and scraping tools
│   ├── code_execution.py      # Code execution tools
│   └── browser_use.py         # Browser Use integration
└── config/
    └── settings.py            # Configuration management
```

## 🚦 Model Support

- **OpenRouter**: Access to 50+ models via unified API
- **OpenAI**: GPT-4, GPT-3.5, etc.
- **Anthropic**: Claude models
- **HuggingFace**: Open source models

## 🔄 Multi-Model Orchestration

For complex tasks requiring multiple specialized models:

```python
from agentic_internet import MultiModelSerpAPISystem

system = MultiModelSerpAPISystem()
result = await system.execute_multi_model_workflow(
    task="Comprehensive market analysis of electric vehicles",
    orchestrator_model="orchestrator"
)
```

## 📊 Performance Tips

1. **Choose the right agent type**:
   - Use `ToolCallingAgent` for API-heavy tasks
   - Use `CodeAgent` for data processing

2. **Optimize tool selection**:
   - Only enable tools you need
   - Use Browser Use for dynamic sites
   - Use simple scrapers for static content

3. **Model selection**:
   - Use faster models for simple tasks
   - Use advanced models for complex reasoning

## 🐛 Troubleshooting

### Missing API Keys
```python
# Check if keys are loaded
from agentic_internet import settings
print(f"OpenRouter: {bool(settings.openrouter_api_key)}")
print(f"Browser Use: {bool(settings.browser_use_api_key)}")
```

### Import Errors for CodeAgent
```python
# Only add installed packages
agent = InternetAgent(
    agent_type="code",
    additional_authorized_imports=["numpy", "pandas"]  # Only installed packages
)
```

## 📚 Resources

- [Browser Use Documentation](https://github.com/browser-use/browser-use-python)
- [SmolagentS Documentation](https://github.com/huggingface/smolagents)
- [OpenRouter API](https://openrouter.ai/docs)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional specialized agents
- More tool integrations
- Enhanced multi-model coordination
- Performance optimizations
