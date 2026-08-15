---
type: component guide
title: Specialized Agents
description: Public domain-specific facades for browser automation, data analysis, content creation, market research, and technical support.
tags: [agents, public-api]
---

# Specialized Agents

`agentic_internet/agents/specialized_agents.py` provides five root-exported subclasses of [InternetAgent](internet-and-research.md). They do not introduce new execution engines: each chooses constructor defaults, creates a detailed prompt, calls inherited `run`, and sometimes wraps the string in metadata. Prompt requests are intentions, not postcondition enforcement.

## Agent contracts

| Class and defaults | Methods and returned shape |
|---|---|
| `BrowserAutomationAgent`, `agent_type="code"`, `max_iterations=20` | `scrape_structured_data(url, data_schema)` requests JSON and returns a decoded dict only when the final text is a JSON object; arrays/scalars/invalid JSON become `{"raw_result": result}`. `fill_form` returns text. `monitor_website` performs one check and returns URL, target, findings, and a suggested interval—it does not schedule monitoring. |
| `DataAnalysisAgent`, code mode | `analyze_dataset(data, analysis_type)` returns `{analysis_type, insights}`. `compare_datasets` turns optional criteria into prompt bullets and returns text. Inputs are string-interpolated without size or schema controls. |
| `ContentCreationAgent`, tool-calling mode | `write_article(topic, style, word_count, sources_required)` returns topic/style/content/target count; it does not verify count or citations. `summarize_content(content, summary_type)` returns text. |
| `MarketResearchAgent`, tool-calling mode, `max_iterations=15` | `analyze_competitor` defaults to products, pricing, position, strengths, and weaknesses and returns `{company, aspects, analysis}`. `market_trends` returns `{industry, timeframe, trends}`. |
| `TechnicalSupportAgent`, code mode | `troubleshoot(problem_description, system_info)` JSON-renders optional system information and returns `{problem, system_info, solution}`. `code_review(code, language, focus_areas)` embeds a language fence and returns language, normalized focus list, and review. |

All inherited failures usually appear inside a result string because `InternetAgent.run` catches exceptions. A metadata wrapper can therefore look structurally successful while its content starts with `Error executing task:`. Code-mode classes inherit the local execution/network implications documented in [System Architecture](../architecture/overview.md).

## Ownership and dependencies

The module depends only on JSON handling and `InternetAgent`; actual models and tools come from the base class. `agentic_internet/__init__.py` and `agents/__init__.py` expose all five classes. `examples/advanced_usage.py` demonstrates selected methods but requires real model/provider configuration and is not a hermetic test.

## Extension recipe

For another domain facade:

1. Subclass `InternetAgent` and set defaults with `kwargs.setdefault` so callers can override them.
2. Keep side effects in registered tools rather than the prompt wrapper.
3. Define whether the method returns raw text or a stable dict. If parsing model output, preserve malformed output explicitly rather than silently discarding it.
4. Export from `agents/__init__.py` and root `agentic_internet/__init__.py` if it is public.
5. Add a focused test with construction bypassed or model/tool mocks, checking prompt-critical behavior and malformed/failure output—not only metadata.

`tests/test_specialized_agents.py` verifies object-versus-nonobject JSON handling for `scrape_structured_data` and article metadata. Constructor defaults and the other helper methods have no focused coverage. Run that file for facade changes and the base/tool suites when changing inherited behavior.