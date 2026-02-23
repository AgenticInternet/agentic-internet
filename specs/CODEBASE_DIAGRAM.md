# Agentic Internet - Codebase Architecture Diagrams

> Generated: 2026-02-23
> Codebase version: 0.1.0

---

## 1. System Context Diagram (Executive View)

High-level view of Agentic Internet, its users, and external dependencies.

```mermaid
flowchart LR
    user["Developer / End User"]
    cli["CLI (typer)"]
    sdk["Python SDK"]

    user --> cli
    user --> sdk

    subgraph agentic_internet["Agentic Internet"]
        cli --> agents["Agent Layer"]
        sdk --> agents
        agents --> tools["Tool Layer"]
        agents --> config["Config & Settings"]
        tools --> config
    end

    subgraph external_llm["LLM Providers (via LiteLLM / OpenRouter)"]
        openrouter["OpenRouter Gateway"]
        openai["OpenAI"]
        anthropic["Anthropic"]
        google["Google (Gemini)"]
        deepseek["DeepSeek"]
        mistral["Mistral AI"]
        xai["xAI (Grok)"]
        qwen["Qwen"]
        perplexity["Perplexity"]
        meta["Meta (Llama)"]
        hf_inference["HuggingFace Inference API"]
    end

    subgraph external_search["Search & Data APIs"]
        serpapi["SerpAPI (Google, Bing, Yahoo, Baidu, Scholar)"]
        ddg["DuckDuckGo Search"]
    end

    subgraph external_browser["Browser Automation"]
        browser_use_cloud["Browser Use Cloud API"]
    end

    subgraph external_mcp["MCP Ecosystem"]
        mcp_stdio["MCP Servers (stdio)"]
        mcp_http["MCP Servers (HTTP)"]
    end

    agents --> openrouter
    agents --> hf_inference
    tools --> serpapi
    tools --> ddg
    tools --> browser_use_cloud
    tools --> mcp_stdio
    tools --> mcp_http
```

### Legend

| Symbol | Meaning |
|--------|---------|
| Rectangle | Internal module or component |
| Rounded rectangle | External system or service |
| Arrow | Runtime dependency / data flow |
| Subgraph | Domain boundary |

### Code Anchors

| Node | Path |
|------|------|
| `cli` | `agentic_internet/cli.py` |
| `agents` | `agentic_internet/agents/` |
| `tools` | `agentic_internet/tools/` |
| `config` | `agentic_internet/config/settings.py` |

---

## 2. Codebase Architecture Diagram (Feature/Domain Level)

Internal architecture showing the major domains and their interconnections.

```mermaid
flowchart TB
    subgraph entry["Entry Points"]
        cli_app["CLI App<br/>(typer)"]
        py_main["__main__.py"]
        pkg_init["Package API<br/>(__init__.py)"]
    end

    subgraph domain_agents["Agent Domain"]
        internet_agent["InternetAgent<br/>(ToolCalling / Code)"]
        research_agent["ResearchAgent"]
        basic_agent["BasicAgent<br/>(Fallback)"]

        subgraph specialized["Specialized Agents"]
            browser_agent["BrowserAutomationAgent"]
            data_agent["DataAnalysisAgent"]
            content_agent["ContentCreationAgent"]
            market_agent["MarketResearchAgent"]
            tech_agent["TechnicalSupportAgent"]
        end

        search_orch["SearchOrchestrator<br/>(Parallel Agent Coordination)"]

        subgraph multi_model["Multi-Model Orchestration"]
            model_manager["ModelManager<br/>(20+ model configs)"]
            mm_system["MultiModelSerpAPISystem"]
            context_eng["ContextEngineeringMixin<br/>(Memory, Context Window)"]
            agent_tool_wrap["AgentTool Wrapper"]
        end
    end

    subgraph domain_tools["Tool Domain"]
        subgraph web_tools["Web Search Tools"]
            web_search["WebSearchTool<br/>(SerpAPI + DDG fallback)"]
            web_scraper["WebScraperTool<br/>(httpx + BeautifulSoup)"]
            news_search["NewsSearchTool"]
        end

        subgraph browser_tools["Browser Automation Tools"]
            bu_sync["BrowserUseTool"]
            bu_async["AsyncBrowserUseTool"]
            bu_structured["StructuredBrowserUseTool"]
        end

        subgraph code_tools["Code Execution Tools"]
            py_executor["PythonExecutorTool<br/>(AST-validated sandbox)"]
            data_analysis["DataAnalysisTool<br/>(pandas operations)"]
        end

        subgraph serpapi_tools["SerpAPI Tools (Multi-Model)"]
            g_search["GoogleSearchTool"]
            g_shopping["GoogleShoppingTool"]
            g_maps["GoogleMapsLocalTool"]
            g_scholar["GoogleScholarTool"]
            multi_engine["MultiEngineSearchTool"]
        end

        subgraph mcp_tools_group["MCP Integration"]
            mcp_integration["MCPToolIntegration"]
            mcp_manager["MCPServerManager"]
            mcp_config["MCPServerConfig"]
            mcp_convenience["mcp_tools() context mgr"]
        end
    end

    subgraph domain_config["Configuration Domain"]
        settings["Settings (Pydantic)"]
        model_config["ModelConfig<br/>(provider catalogs)"]
        agent_config["AgentConfig"]
        tool_config["ToolConfig"]
    end

    subgraph domain_utils["Utilities"]
        model_utils["model_utils.py<br/>(initialize_model, factory)"]
        exceptions["exceptions.py<br/>(error hierarchy)"]
    end

    %% Entry point flows
    cli_app --> internet_agent
    cli_app --> research_agent
    cli_app --> mm_system
    cli_app --> mcp_convenience
    py_main --> cli_app
    pkg_init --> internet_agent
    pkg_init --> specialized

    %% Agent internal flows
    internet_agent --> web_tools
    internet_agent --> browser_tools
    internet_agent --> code_tools
    research_agent -.-> internet_agent
    specialized -.-> internet_agent

    search_orch --> internet_agent
    mm_system --> model_manager
    mm_system --> context_eng
    mm_system --> serpapi_tools
    mm_system --> agent_tool_wrap

    %% Config flows
    internet_agent --> model_utils
    model_utils --> settings
    settings --> model_config
    settings --> agent_config
    settings --> tool_config

    %% Tool dependencies
    mcp_integration --> mcp_manager
    mcp_manager --> mcp_config
```

### Legend

| Symbol | Meaning |
|--------|---------|
| Solid arrow (`-->`) | Direct runtime call / dependency |
| Dashed arrow (`-.->`) | Inheritance relationship |
| Subgraph | Feature domain boundary |
| `<br/>` in labels | Supplementary info about the node |

### Code Anchors

| Node | Path |
|------|------|
| `cli_app` | `agentic_internet/cli.py` |
| `py_main` | `agentic_internet/__main__.py` |
| `pkg_init` | `agentic_internet/__init__.py` |
| `internet_agent` | `agentic_internet/agents/internet_agent.py:28` (class InternetAgent) |
| `research_agent` | `agentic_internet/agents/internet_agent.py:254` (class ResearchAgent) |
| `basic_agent` | `agentic_internet/agents/basic_agent.py:12` (class BasicAgent) |
| `browser_agent` | `agentic_internet/agents/specialized_agents.py:13` |
| `data_agent` | `agentic_internet/agents/specialized_agents.py:100` |
| `content_agent` | `agentic_internet/agents/specialized_agents.py:173` |
| `market_agent` | `agentic_internet/agents/specialized_agents.py:243` |
| `tech_agent` | `agentic_internet/agents/specialized_agents.py:325` |
| `search_orch` | `agentic_internet/agents/search_orchestrator.py:119` (class SearchOrchestrator) |
| `model_manager` | `agentic_internet/agents/multi_model_serpapi.py:32` (class ModelManager) |
| `mm_system` | `agentic_internet/agents/multi_model_serpapi.py:914` (class MultiModelSerpAPISystem) |
| `context_eng` | `agentic_internet/agents/multi_model_serpapi.py:783` (class ContextEngineeringMixin) |
| `agent_tool_wrap` | `agentic_internet/agents/multi_model_serpapi.py:762` (class AgentTool) |
| `web_search` | `agentic_internet/tools/web_search.py:110` |
| `web_scraper` | `agentic_internet/tools/web_search.py:184` |
| `news_search` | `agentic_internet/tools/web_search.py:238` |
| `bu_sync` | `agentic_internet/tools/browser_use.py:21` |
| `bu_async` | `agentic_internet/tools/browser_use.py:74` |
| `bu_structured` | `agentic_internet/tools/browser_use.py:150` |
| `py_executor` | `agentic_internet/tools/code_execution.py:162` |
| `data_analysis` | `agentic_internet/tools/code_execution.py:259` |
| `g_search` | `agentic_internet/agents/multi_model_serpapi.py:390` |
| `g_shopping` | `agentic_internet/agents/multi_model_serpapi.py:470` |
| `g_maps` | `agentic_internet/agents/multi_model_serpapi.py:540` |
| `g_scholar` | `agentic_internet/agents/multi_model_serpapi.py:593` |
| `multi_engine` | `agentic_internet/agents/multi_model_serpapi.py:646` |
| `mcp_integration` | `agentic_internet/tools/mcp_integration.py:53` |
| `mcp_manager` | `agentic_internet/tools/mcp_integration.py:255` |
| `mcp_config` | `agentic_internet/tools/mcp_integration.py:216` |
| `mcp_convenience` | `agentic_internet/tools/mcp_integration.py:402` |
| `settings` | `agentic_internet/config/settings.py:161` (class Settings) |
| `model_config` | `agentic_internet/config/settings.py:15` (class ModelConfig) |
| `agent_config` | `agentic_internet/config/settings.py:139` |
| `tool_config` | `agentic_internet/config/settings.py:150` |
| `model_utils` | `agentic_internet/utils/model_utils.py` |
| `exceptions` | `agentic_internet/exceptions.py` |

---

## 3. Agent Inheritance & Type Hierarchy

Shows how the different agent types relate to each other.

```mermaid
flowchart TB
    smolagents_tca["smolagents.ToolCallingAgent"]
    smolagents_ca["smolagents.CodeAgent"]
    smolagents_tool["smolagents.Tool"]

    internet_agent["InternetAgent<br/>(wraps ToolCallingAgent or CodeAgent)"]
    research_agent["ResearchAgent"]
    browser_agent["BrowserAutomationAgent"]
    data_agent["DataAnalysisAgent"]
    content_agent["ContentCreationAgent"]
    market_agent["MarketResearchAgent"]
    tech_agent["TechnicalSupportAgent"]
    basic_agent["BasicAgent<br/>(standalone, no smolagents base)"]

    smolagents_tca -.-> internet_agent
    smolagents_ca -.-> internet_agent
    internet_agent -.-> research_agent
    internet_agent -.-> browser_agent
    internet_agent -.-> data_agent
    internet_agent -.-> content_agent
    internet_agent -.-> market_agent
    internet_agent -.-> tech_agent

    search_orch["SearchOrchestrator<br/>(manages SearchAgentWrappers)"]
    search_wrapper["SearchAgentWrapper<br/>(wraps any smolagents agent)"]
    search_orch --> search_wrapper

    mm_system["MultiModelSerpAPISystem<br/>(ContextEngineeringMixin)"]
    agent_tool["AgentTool<br/>(extends smolagents.Tool)"]
    smolagents_tool -.-> agent_tool
    mm_system --> agent_tool
```

### Legend

| Symbol | Meaning |
|--------|---------|
| Dashed arrow (`-.->`) | Inheritance (extends / wraps) |
| Solid arrow (`-->`) | Composition (creates / manages) |

---

## 4. Multi-Model Orchestration Flow

Shows how the `MultiModelSerpAPISystem` coordinates multiple LLM models and worker agents.

```mermaid
flowchart LR
    subgraph user_input["User Input"]
        task["Task String"]
    end

    subgraph orchestration["MultiModelSerpAPISystem"]
        mm["MultiModelSerpAPISystem"]
        ctx["ContextEngineeringMixin<br/>(Memory + Context Window)"]
        mgr["ModelManager<br/>(20+ LLM configs)"]

        mm --> ctx
        mm --> mgr
    end

    subgraph workers["Specialized Worker Agents"]
        w_search["search_researcher<br/>(Sonar Pro)"]
        w_ecom["ecommerce_analyst<br/>(Gemini Pro)"]
        w_local["local_business_analyst<br/>(Mistral Large)"]
        w_acad["academic_researcher<br/>(Sonar Reasoning)"]
    end

    subgraph orchestrator_agent["Orchestrator Agent"]
        orch_code["CodeAgent<br/>(Claude Opus 4.5)"]
    end

    subgraph serpapi["SerpAPI Tools"]
        st_google["GoogleSearchTool"]
        st_shop["GoogleShoppingTool"]
        st_maps["GoogleMapsLocalTool"]
        st_scholar["GoogleScholarTool"]
        st_multi["MultiEngineSearchTool"]
    end

    subgraph result_output["Output"]
        result["JSON Result<br/>(primary + performance + cross-engine)"]
    end

    task --> mm
    mgr --> |"assigns models"| workers
    mgr --> |"assigns model"| orch_code
    mm --> |"wraps as AgentTool"| workers
    mm --> |"provides SerpAPI tools"| orch_code
    workers -.-> |"wrapped as tools"| orch_code
    orch_code --> |"delegates subtasks"| workers
    workers --> serpapi
    orch_code --> result
```

### Legend

| Symbol | Meaning |
|--------|---------|
| Solid arrow | Data flow / runtime call |
| Dashed arrow | Tool delegation |
| Parenthetical text | Default model assignment |

### Code Anchors

| Node | Path |
|------|------|
| `mm` | `agentic_internet/agents/multi_model_serpapi.py:914` |
| `mgr` | `agentic_internet/agents/multi_model_serpapi.py:32` |
| `ctx` | `agentic_internet/agents/multi_model_serpapi.py:783` |
| Worker creation | `agentic_internet/agents/multi_model_serpapi.py:1120` (`setup_multi_model_workers`) |
| Orchestration entry | `agentic_internet/agents/multi_model_serpapi.py:1178` (`execute_multi_model_workflow`) |

---

## 5. Request Lifecycle (Single Agent Task)

Shows the flow from CLI invocation through agent execution to result.

```mermaid
sequenceDiagram
    participant User
    participant CLI as CLI (typer)
    participant Agent as InternetAgent
    participant ModelUtils as model_utils
    participant Settings as Settings
    participant LLM as LLM Provider<br/>(via LiteLLM)
    participant Tools as Tool Layer
    participant External as External APIs

    User->>CLI: agentic-internet run "task"
    CLI->>Settings: Load .env + config
    CLI->>Agent: InternetAgent(model_id, tools)
    Agent->>ModelUtils: initialize_model(model_id)
    ModelUtils->>Settings: get_model_provider()
    ModelUtils->>Settings: get_api_key_for_provider()
    ModelUtils-->>Agent: LiteLLMModel instance

    Agent->>Agent: _get_default_tools()
    Agent->>Agent: _create_agent() [ToolCalling or Code]

    Agent->>LLM: agent.run(task)
    loop Agent reasoning loop (max_iterations)
        LLM-->>Agent: Tool call decision
        Agent->>Tools: Execute tool (e.g., web_search)
        Tools->>External: API call (SerpAPI / DDG / BrowserUse)
        External-->>Tools: Raw results
        Tools-->>Agent: Formatted results
        Agent->>LLM: Results + next step
    end
    LLM-->>Agent: Final answer
    Agent-->>CLI: Result string
    CLI-->>User: Display via Rich panel
```

### Code Anchors

| Step | Path |
|------|------|
| CLI `run` command | `agentic_internet/cli.py:61` |
| `InternetAgent.__init__` | `agentic_internet/agents/internet_agent.py:33` |
| `initialize_model` | `agentic_internet/utils/model_utils.py:60` |
| `_get_default_tools` | `agentic_internet/agents/internet_agent.py:112` |
| `InternetAgent.run` | `agentic_internet/agents/internet_agent.py:148` |

---

## 6. Tool Selection & Fallback Strategy

Shows how the web search tools use SerpAPI with DuckDuckGo fallback.

```mermaid
flowchart TB
    query["Search Query"]
    query --> check_serpapi{"SerpAPI key<br/>available?"}

    check_serpapi -->|Yes| try_serpapi["Call SerpAPI<br/>(GoogleSearch)"]
    check_serpapi -->|No| ddg["Call DuckDuckGo<br/>(DDGS)"]

    try_serpapi --> serpapi_ok{"Results<br/>returned?"}
    serpapi_ok -->|Yes| format_serp["Format SerpAPI Results"]
    serpapi_ok -->|No| ddg

    ddg --> ddg_ok{"Results<br/>returned?"}
    ddg_ok -->|Yes| format_ddg["Format DDG Results"]
    ddg_ok -->|No| no_results["No search results found"]

    format_serp --> result["Return formatted results"]
    format_ddg --> result
```

### Code Anchors

| Node | Path |
|------|------|
| `_search_with_fallback` | `agentic_internet/tools/web_search.py:99` |
| `_search_serpapi` | `agentic_internet/tools/web_search.py:26` |
| `_search_ddgs` | `agentic_internet/tools/web_search.py:66` |

---

## 7. MCP Integration Architecture

Shows how MCP servers connect to the agent system.

```mermaid
flowchart LR
    subgraph user_code["User Code / CLI"]
        mcp_ctx["mcp_tools()<br/>context manager"]
        cli_mcp["CLI mcp run"]
    end

    subgraph mcp_layer["MCP Integration Layer"]
        mcp_int["MCPToolIntegration"]
        mcp_mgr["MCPServerManager"]
        mcp_cfg["MCPServerConfig"]
        env_loader["load_mcp_config_from_env()"]
    end

    subgraph smolagents_mcp["smolagents"]
        tool_collection["ToolCollection.from_mcp()"]
    end

    subgraph mcp_servers["MCP Servers"]
        stdio_server["Local Server<br/>(stdio subprocess)"]
        http_server["Remote Server<br/>(streamable-http)"]
    end

    subgraph agent_layer["Agent Layer"]
        agent["InternetAgent"]
    end

    cli_mcp --> mcp_ctx
    mcp_ctx --> mcp_int
    mcp_int --> tool_collection
    mcp_mgr --> mcp_int
    mcp_cfg --> mcp_mgr
    env_loader --> mcp_cfg
    tool_collection --> stdio_server
    tool_collection --> http_server
    tool_collection --> |"yields tools"| agent
```

### Code Anchors

| Node | Path |
|------|------|
| `mcp_ctx` | `agentic_internet/tools/mcp_integration.py:402` |
| `mcp_int` | `agentic_internet/tools/mcp_integration.py:53` |
| `mcp_mgr` | `agentic_internet/tools/mcp_integration.py:255` |
| `mcp_cfg` | `agentic_internet/tools/mcp_integration.py:216` |
| `env_loader` | `agentic_internet/tools/mcp_integration.py:453` |
| CLI `mcp run` | `agentic_internet/cli.py:718` |

---

## 8. Configuration & Provider Resolution

How model provider is resolved and API keys are matched.

```mermaid
flowchart TB
    model_id["model_id string"]
    model_id --> provider_check{"Explicit<br/>provider prefix?"}

    provider_check -->|"openrouter/"| openrouter["Provider: openrouter"]
    provider_check -->|"gpt-*, o3, o4"| openai["Provider: openai"]
    provider_check -->|"claude-*"| anthropic["Provider: anthropic"]
    provider_check -->|"meta-llama/*"| huggingface["Provider: huggingface"]
    provider_check -->|Unknown| auto_detect{"Auto-detect:<br/>check API keys"}

    auto_detect -->|"OPENROUTER_API_KEY set"| openrouter
    auto_detect -->|"OPENAI_API_KEY set"| openai
    auto_detect -->|"ANTHROPIC_API_KEY set"| anthropic
    auto_detect -->|"HUGGINGFACE_TOKEN set"| huggingface
    auto_detect -->|"No keys"| fail["No provider found"]

    openrouter --> create_litellm["LiteLLMModel(model_id, api_key)"]
    openai --> create_litellm
    anthropic --> create_litellm
    huggingface --> create_hf["InferenceClientModel(model_id, token)"]

    create_litellm --> model_instance["Model Instance"]
    create_hf --> model_instance
    fail --> fallback["get_any_available_model()"]
    fallback --> model_instance
```

### Code Anchors

| Node | Path |
|------|------|
| `get_model_provider` | `agentic_internet/config/settings.py:208` |
| `_create_model_for_provider` | `agentic_internet/utils/model_utils.py:19` |
| `initialize_model` | `agentic_internet/utils/model_utils.py:60` |
| `get_any_available_model` | `agentic_internet/utils/model_utils.py:95` |

---

## 9. CLI Command Map

Overview of all CLI commands and their relationships to internal components.

```mermaid
flowchart TB
    cli_root["agentic-internet"]

    cli_root --> cmd_chat["chat<br/>Interactive session"]
    cli_root --> cmd_run["run<br/>Single task"]
    cli_root --> cmd_research["research<br/>Topic research"]
    cli_root --> cmd_multi["multi<br/>Multi-model task"]
    cli_root --> cmd_orchestrate["orchestrate<br/>Complex orchestration"]
    cli_root --> cmd_tools["tools<br/>List tools"]
    cli_root --> cmd_news["news<br/>News search"]
    cli_root --> cmd_models["models<br/>Model catalog"]
    cli_root --> cmd_config["config<br/>Show config"]
    cli_root --> cmd_version["version"]

    subgraph mcp_sub["mcp subcommand"]
        cmd_mcp_list["mcp list"]
        cmd_mcp_info["mcp info"]
        cmd_mcp_run["mcp run"]
        cmd_mcp_test["mcp test"]
    end
    cli_root --> mcp_sub

    cmd_chat --> internet_agent["InternetAgent"]
    cmd_run --> internet_agent
    cmd_research --> research_agent["ResearchAgent"]
    cmd_multi --> mm_system["MultiModelSerpAPISystem"]
    cmd_orchestrate --> mm_system
    cmd_news --> mm_system
    cmd_mcp_run --> mcp_tools["mcp_tools()"]
    mcp_tools --> internet_agent
```

### Code Anchors

| Command | Path |
|---------|------|
| `chat` | `agentic_internet/cli.py:30` |
| `run` | `agentic_internet/cli.py:61` |
| `research` | `agentic_internet/cli.py:104` |
| `multi` | `agentic_internet/cli.py:196` |
| `orchestrate` | `agentic_internet/cli.py:293` |
| `tools` | `agentic_internet/cli.py:364` |
| `news` | `agentic_internet/cli.py:410` |
| `models` | `agentic_internet/cli.py:503` |
| `config` | `agentic_internet/cli.py:160` |
| `version` | `agentic_internet/cli.py:885` |
| `mcp list` | `agentic_internet/cli.py:627` |
| `mcp info` | `agentic_internet/cli.py:676` |
| `mcp run` | `agentic_internet/cli.py:718` |
| `mcp test` | `agentic_internet/cli.py:814` |

---

## 10. Exception Hierarchy

```mermaid
flowchart TB
    base["AgenticInternetError"]
    base --> model_init["ModelInitializationError"]
    base --> provider_nf["ProviderNotFoundError"]
    base --> api_key["APIKeyMissingError"]
    base --> search_err["SearchError"]
    base --> tool_exec["ToolExecutionError"]
    base --> config_err["ConfigurationError"]
    base --> mcp_err["MCPError"]

    tool_exec --> code_exec["CodeExecutionError"]
    tool_exec --> browser_err["BrowserAutomationError"]
    code_exec --> unsafe["UnsafeCodeError"]
    mcp_err --> mcp_na["MCPNotAvailableError"]
```

### Code Anchors

| Exception | Path |
|-----------|------|
| All exceptions | `agentic_internet/exceptions.py` |

---

## Unknowns / Assumptions

| Item | Status | Verification |
|------|--------|-------------|
| No database or persistent storage detected | Confirmed | All state is in-memory (AgentMemory, ContextWindow). No DB drivers in dependencies. |
| No message queue or event bus | Confirmed | ThreadPoolExecutor used for parallel agents; asyncio for async flows. No Kafka/Redis/SQS. |
| smolagents is the core framework | Confirmed | All agents wrap `ToolCallingAgent`, `CodeAgent`, or `BasicAgent` from smolagents. |
| OpenRouter is the primary LLM gateway | Assumed | Default model is `openrouter/anthropic/claude-opus-4.5`. Most model configs route through OpenRouter. Direct provider keys are fallbacks. |
| SerpAPI tools live in `multi_model_serpapi.py` rather than `tools/` | Confirmed | Google Search/Shopping/Maps/Scholar/MultiEngine tools are defined alongside the multi-model system, not in the tools package. This is an architectural coupling. |
| No authentication or user management | Confirmed | CLI is single-user. No auth middleware or user model. |
| Examples directory is documentation-only | Confirmed | `agentic_internet/examples/` contains usage examples, not tested application code. |
