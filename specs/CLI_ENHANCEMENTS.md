# CLI Enhancements for Multi-Model Orchestration

## Overview

The Agentic Internet CLI has been enhanced to support multi-model orchestration, news analysis, and improved model management. These enhancements provide powerful new capabilities for running tasks across multiple AI models simultaneously.

## New Commands

### 1. **multi** - Multi-Model Task Execution
Run tasks using multiple AI models concurrently.

```bash
agentic-internet multi "Analyze the impact of AI on healthcare" \
  --models gpt-4 \
  --models claude-3-opus \
  --workers 3 \
  --output results.json
```

**Options:**
- `--models, -m`: Specify models to use (can be used multiple times)
- `--verbose/--quiet, -v/-q`: Control output verbosity
- `--output, -o`: Save results to file
- `--news, -n`: Include news research models
- `--workers, -w`: Maximum concurrent workers (default: 3)

### 2. **orchestrate** - Complex Task Orchestration
Execute complex tasks using a coordinator model to manage worker models.

```bash
agentic-internet orchestrate "Create a comprehensive market analysis" \
  --coordinator gpt-4 \
  --workers gpt-3.5-turbo \
  --workers claude-3-sonnet \
  --output analysis.json
```

**Options:**
- `--coordinator, -c`: Model to use as coordinator (default: gpt-4)
- `--workers, -w`: Worker models (can specify multiple)
- `--verbose/--quiet`: Control output verbosity
- `--output, -o`: Save results to file

### 3. **news** - News Search and Analysis
Search and analyze news articles on specific topics.

```bash
agentic-internet news "artificial intelligence breakthroughs" \
  --time 7d \
  --sources "TechCrunch" \
  --sources "Wired" \
  --limit 20 \
  --format markdown \
  --output news_report.md
```

**Options:**
- `--sources, -s`: Specific news sources to search
- `--time, -t`: Time frame (1h, 24h, 7d, 30d)
- `--limit, -l`: Maximum number of results
- `--output, -o`: Save results to file
- `--format, -f`: Output format (markdown, json, text)

### 4. **models** - Model Management
List and manage available AI models.

```bash
# List all models
agentic-internet models --list

# Show models by category
agentic-internet models --category news

# Show detailed information
agentic-internet models --details
```

**Options:**
- `--list, -l`: List all available models
- `--category, -c`: Filter by category (general, news, research, code)
- `--details, -d`: Show detailed model information

## Enhanced Commands

### **tools** Command Enhancement
The tools command now supports multi-model tool listing:

```bash
# Show regular tools
agentic-internet tools --list

# Show multi-model SerpAPI tools
agentic-internet tools --multi
```

## Available Models

The system now supports various model categories:

### General Models
- **gpt-4** (OpenAI): Most capable GPT-4 model
- **gpt-3.5-turbo** (OpenAI): Fast and efficient
- **claude-3-opus** (Anthropic): Claude's most capable model
- **claude-3-sonnet** (Anthropic): Balanced performance
- **meta-llama/Llama-3.3-70B-Instruct** (Meta): Open source LLM

### News Models
- **gpt-4-news** (OpenAI): Optimized for news analysis
- **claude-news-analyst** (Anthropic): News and media analysis

### Research Models
- **gpt-4-research** (OpenAI): Academic research model
- **claude-researcher** (Anthropic): Deep research capabilities

### Code Models
- **gpt-4-code** (OpenAI): Code generation and analysis
- **codellama-34b** (Meta): Specialized code model

## Multi-Model SerpAPI Tools

The following specialized tools are available for multi-model operations:

1. **GoogleSearchTool**: Web search using Google
2. **GoogleShoppingTool**: Product and price searches
3. **GoogleMapsLocalTool**: Local business and place searches
4. **GoogleScholarTool**: Academic paper and citation searches
5. **MultiEngineSearchTool**: Cross-engine search capabilities

## Usage Examples

### Example 1: Multi-Model Research
```bash
agentic-internet multi "Research quantum computing applications" \
  --models gpt-4 \
  --models claude-3-opus \
  --verbose \
  --output quantum_research.json
```

### Example 2: News Analysis
```bash
agentic-internet news "climate change policy updates" \
  --time 24h \
  --limit 15 \
  --format markdown \
  --output climate_news.md
```

### Example 3: Orchestrated Task
```bash
agentic-internet orchestrate "Develop a business plan for an AI startup" \
  --coordinator gpt-4 \
  --workers claude-3-sonnet \
  --workers gpt-3.5-turbo \
  --output business_plan.json
```

### Example 4: Model Information
```bash
# List all research models with details
agentic-internet models --category research --details
```

## Configuration

Ensure the following environment variables are set:
- `SERP_API_KEY`: Required for SerpAPI functionality
- `OPENAI_API_KEY`: Required for OpenAI models
- `ANTHROPIC_API_KEY`: Required for Claude models
- `HUGGINGFACE_API_KEY`: Required for open-source models

## Async Execution

The multi-model commands leverage asynchronous execution for improved performance:
- Concurrent model processing
- Event loop management
- Progress indicators with spinners
- Proper error handling and recovery

## Output Formats

All new commands support multiple output formats:
- **JSON**: Structured data for programmatic use
- **Markdown**: Formatted documents for readability
- **Text**: Plain text output
- **Panel Display**: Rich terminal output with formatting

## Error Handling

Enhanced error handling includes:
- Clear error messages with context
- Warnings for missing API keys
- Graceful degradation when models are unavailable
- Detailed stack traces in verbose mode

## Future Enhancements

Planned improvements include:
- Model performance benchmarking
- Cost tracking and optimization
- Custom model configurations
- Advanced orchestration strategies
- Real-time streaming responses
- Model-specific parameter tuning
