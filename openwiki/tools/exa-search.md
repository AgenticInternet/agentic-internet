---
type: tool guide
title: Exa Search
description: Semantic search and find-similar tool schemas, content controls, client lifecycle, and conditional registration.
tags: [tools, search, exa]
---

# Exa Search

`agentic_internet/tools/exa_search.py` adapts `exa_py` into `ExaSearchTool` (`exa_search`) and `ExaFindSimilarTool` (`exa_find_similar`). Both return formatted strings rather than structured objects and are exported from `agentic_internet.tools`.

## Normalization and request flow

`ExaResult` is the internal normalized dataclass: title, URL, optional text, summary, author, publish date, score, and highlights. `from_sdk` tolerates missing fields. `snippet(max_chars)` prefers summary, joined highlights, text, then `No description`, with ellipsis truncation.

`ExaSearchTool.forward` checks SDK/key availability, creates a new `Exa(api_key)` client, adds `x-exa-integration: agentic-internet`, builds content controls, normalizes optional search type/category allowlists, adds domain/date filters, calls `search_and_contents`, normalizes `response.results`, and formats ranked text. Unknown search types/categories are omitted so SDK defaults apply. `ExaFindSimilarTool` follows the same content path around `find_similar_and_contents(url=...)` and can exclude the source domain.

Constructor knobs control result count, text, highlights, summaries, query-focused summaries, and text character limit. Explicit zero count/limit values fall back to defaults because constructors use `or`. There is no alternate provider fallback, retry, close protocol, cache, or backoff; SDK exceptions become strings containing provider detail.

## Registration and boundaries

[InternetAgent](../agents/internet-and-research.md) adds both tools only when web search is enabled and `settings.exa_api_key` is truthy. Registration checks the import-time settings field, but tool constructors reread `os.getenv("EXA_API_KEY")`; mutating only settings can therefore register unavailable tool instances. K-LLM bundle declarations mention Exa names, but its inventory does not instantiate these tools.

Queries, domain lists, dates, and source URLs lack local structural validation. In particular, find-similar does not reuse the web scraper’s URL syntax validation. Returned exception text may expose SDK/server details.

## Extension and tests

Use `_build_contents_kwargs`, normalizers, `ExaResult`, and `_format_results` rather than duplicating provider adaptation. A public change may require tool export, default-agent gate, K-LLM inventory/bundle wiring, sample environment placeholder, and enabled/disabled tests.

`tests/test_exa_search.py` covers field defaults, snippet preference/truncation, formatting, exact content kwargs, normalizers, key/SDK absence, integration header and filter forwarding, unknown type omission, errors, similar-page arguments, and default registration names. It does not cover constructor environment reads, date/domain/URL validity, zero options, client cleanup, or all find-similar failure branches.