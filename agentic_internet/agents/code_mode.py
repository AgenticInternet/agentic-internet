"""
Code-mode agent factory with ToolFacade.

Provides ``create_code_mode_agent()``: a factory that wraps any list of
smolagents Tools behind a single ``ToolFacade`` object (``api``), then
constructs a ``smolagents.CodeAgent`` whose system prompt teaches it to
discover tools via ``api.search()`` and call them with plain Python.

This solves two problems with the default ``ToolCallingAgent`` + MCP path:

1. **Tool-list explosion** - large MCP servers expose dozens of tools; putting
   them all in the agent's tool list overwhelms the context window and causes
   poor selection.  The facade collapses them into one ``api`` object.

2. **Sequential calls only** - ``ToolCallingAgent`` invokes tools one at a
   time.  ``CodeAgent`` can write loops, pipelines and data transformations
   in a single step.

Usage::

    from agentic_internet.agents.code_mode import create_code_mode_agent
    from smolagents import DuckDuckGoSearchTool, VisitWebpageTool

    agent = create_code_mode_agent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    )
    result = agent.run("Research deer-flow by ByteDance and summarise the top 3 features.")
    print(result)

With MCP tools (keep the context manager open during the run)::

    from agentic_internet.tools.mcp_integration import mcp_tools

    with mcp_tools(server_path="./my_server.py", trust_remote_code=True) as tools:
        agent = create_code_mode_agent(tools=list(tools))
        result = agent.run("Use the MCP tools to accomplish this task")
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

from smolagents import CodeAgent, Tool

from ..exceptions import ModelInitializationError
from ..utils.model_utils import initialize_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Imports always available inside ExecuteTool's sandbox.
DEFAULT_EXECUTE_IMPORTS: list[str] = ["json", "datetime", "re", "math", "os"]

#: Tool names reserved by the facade/meta-tool layer; MCP tools must not use these.
_RESERVED_NAMES: frozenset[str] = frozenset({"search", "execute", "_tools"})


# ---------------------------------------------------------------------------
# ToolFacade
# ---------------------------------------------------------------------------


class ToolFacade:
    """Wraps a flat list of smolagents Tools behind a single ``api`` object.

    The agent can:

    * Discover tools with ``api.search("keyword")`` — returns formatted
      signatures for tools whose name or description contains the keyword
      (case-insensitive substring match).
    * Call a tool with ``api.tool_name(arg1=val1, ...)`` — delegates directly
      to ``tool.forward()``.

    For tool names that are not valid Python identifiers (e.g. ``get-weather``)
    use the escape hatch ``api._tools["get-weather"](...)``.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in _RESERVED_NAMES:
                logger.warning(
                    "Tool name '%s' is reserved by ToolFacade and will be skipped. Rename the tool to avoid shadowing.",
                    tool.name,
                )
                continue
            self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        """Return formatted signatures for tools matching *query*.

        The search is a case-insensitive substring match against both the tool
        name and its description.

        Returns a newline-separated list of::

            api.tool_name(param: type, ...)  # description (first 120 chars)

        or ``"No matching tools found."`` if there are no matches.
        """
        q = query.lower()
        matches: list[str] = []
        for name, tool in self._tools.items():
            if q not in name.lower() and q not in tool.description.lower():
                continue

            # Build parameter signature
            params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in (tool.inputs or {}).items())

            call = f"api.{name}({params})" if name.isidentifier() else f'api._tools["{name}"]({params})'

            matches.append(f"{call}  # {tool.description[:120]}")

        return "\n".join(matches) if matches else "No matching tools found."

    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion for dunder / private attributes
        if name.startswith("_"):
            raise AttributeError(name)
        tools = object.__getattribute__(self, "_tools")
        if name in tools:
            return tools[name]
        raise AttributeError(
            f"Tool '{name}' not found in the facade. Use api.search('{name}') to discover available tools."
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ToolFacade tools={list(self._tools.keys())}>"


# ---------------------------------------------------------------------------
# SearchTool
# ---------------------------------------------------------------------------


class SearchTool(Tool):
    """Discover tools by keyword.

    Always call this first when you don't know the exact tool name or its
    argument signature.  Pass a keyword related to what you want to do.
    """

    name = "search"
    description = (
        "Search the entire tool registry by keyword. "
        "Always start here when you don't know exact tool names or signatures. "
        "Returns callable signatures you can use with the api object."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "query": {
            "type": "string",
            "description": "Keyword to search for (tool name or description words).",
        }
    }
    output_type = "string"

    def __init__(self, facade: ToolFacade | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.facade = facade

    def forward(self, query: str) -> str:
        if self.facade is None:
            return "Tool facade not initialised."
        return self.facade.search(query)

    # ------------------------------------------------------------------
    # Serialisation safety for E2B
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Temporarily hide the non-serialisable facade during E2B serialisation."""
        facade_backup = self.__dict__.pop("facade", None)
        try:
            return super().to_dict()  # type: ignore[no-any-return]
        finally:
            if facade_backup is not None:
                self.facade = facade_backup


# ---------------------------------------------------------------------------
# ExecuteTool
# ---------------------------------------------------------------------------


class ExecuteTool(Tool):
    """Execute Python code with the full tool API available as ``api``.

    Write loops, conditionals, data processing, and multiple tool calls in
    **one step**.  Use ``print(...)`` for intermediate debug output.

    Example::

        results = []
        for query in ["Python", "Rust", "Go"]:
            r = api.web_search(query=query)
            results.append(r)
        print(results)
    """

    name = "execute"
    description = (
        "Execute Python code with the complete API available as `api`. "
        "Write loops, conditionals, data processing, multiple tool calls - "
        "anything you need in ONE step. Use print(...) for debug output."
    )
    inputs: ClassVar[dict[str, Any]] = {
        "code": {
            "type": "string",
            "description": "Valid Python code. The `api` object and `json` module are pre-imported.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        facade: ToolFacade | None = None,
        additional_authorized_imports: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.facade = facade
        self._additional_imports = additional_authorized_imports or []

    def forward(self, code: str) -> str:
        import json as _json
        import logging as _logging

        from smolagents import LocalPythonExecutor as _LocalPythonExecutor

        all_imports = list(dict.fromkeys([*DEFAULT_EXECUTE_IMPORTS, *self._additional_imports]))

        executor = _LocalPythonExecutor(
            additional_authorized_imports=all_imports,
            max_print_outputs_length=15_000,
        )
        # static_tools are the callable names available inside the sandboxed code.
        # We expose: api (the facade), json, and print.
        executor.static_tools = {
            "api": self.facade,
            "json": _json,
            "print": print,
        }
        # Also seed state so that any subsequent calls on the same executor instance
        # have access to these names via variable lookup.
        executor.state["api"] = self.facade
        executor.state["json"] = _json

        try:
            code_output = executor(code)
            # code_output.output is the last expression value; logs are print() output
            logs = code_output.logs.strip() if code_output.logs else ""
            output = code_output.output
            if output is not None:
                return str(output)
            if logs:
                return logs
            return "Execution successful."
        except Exception as exc:
            _logging.getLogger(__name__).debug("ExecuteTool error", exc_info=True)
            return f"Execution error: {exc}"

    # ------------------------------------------------------------------
    # Serialisation safety for E2B
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Temporarily hide the non-serialisable facade during E2B serialisation."""
        facade_backup = self.__dict__.pop("facade", None)
        try:
            return super().to_dict()  # type: ignore[no-any-return]
        finally:
            if facade_backup is not None:
                self.facade = facade_backup


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_code_mode_agent(
    tools: list[Tool],
    model_id: str | None = None,
    verbosity_level: int = 2,
    max_steps: int = 25,
    executor_type: str = "local",
    additional_authorized_imports: list[str] | None = None,
    **kwargs: Any,
) -> CodeAgent:
    """Create a ``CodeAgent`` that accesses all *tools* through a ``ToolFacade``.

    Args:
        tools: Tools to expose through the facade (MCP tools, default tools,
            or any combination).  Pass an empty list for a bare agent.
        model_id: LLM to use.  Falls back to ``settings.model.name`` ->
            provider auto-detection via ``initialize_model()``.
        verbosity_level: smolagents verbosity (0-2).  2 = full output.
        max_steps: Maximum reasoning steps before the agent stops.
        executor_type: ``"local"`` (default) or ``"e2b"`` for E2B sandbox.
            Falls back to ``"local"`` with a warning when
            ``E2B_API_KEY`` is not set.
        additional_authorized_imports: Extra Python modules available inside
            ``ExecuteTool``'s sandbox.  Added on top of
            ``DEFAULT_EXECUTE_IMPORTS``.
        **kwargs: Forwarded to ``CodeAgent.__init__()``.

    Returns:
        Configured ``CodeAgent`` with ``search`` and ``execute`` meta-tools.

    Raises:
        ModelInitializationError: If no model can be initialised.
    """
    # ------------------------------------------------------------------
    # Build facade
    # ------------------------------------------------------------------
    facade = ToolFacade(tools)

    # ------------------------------------------------------------------
    # Meta-tools
    # ------------------------------------------------------------------
    search_tool = SearchTool(facade=facade)
    execute_tool = ExecuteTool(
        facade=facade,
        additional_authorized_imports=additional_authorized_imports,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = initialize_model(model_id)
    if model is None:
        raise ModelInitializationError(
            model_id=model_id or "default",
            cause="No model could be initialised. Check your API keys.",
        )

    # ------------------------------------------------------------------
    # Executor type: fall back to local if E2B key is missing
    # ------------------------------------------------------------------
    resolved_executor_type = executor_type
    executor_kwargs: dict[str, Any] = {}

    if executor_type == "e2b":
        e2b_api_key = os.environ.get("E2B_API_KEY")
        if not e2b_api_key:
            logger.warning("E2B_API_KEY is not set; falling back to executor_type='local'.")
            resolved_executor_type = "local"
        else:
            executor_kwargs["api_key"] = e2b_api_key

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    # add_base_tools defaults to False: base tools (DuckDuckGo, etc.) require
    # optional packages that may not be installed.  Callers can override via kwargs.
    kwargs.setdefault("add_base_tools", False)

    agent = CodeAgent(
        tools=[search_tool, execute_tool],
        model=model,
        verbosity_level=verbosity_level,
        max_steps=max_steps,
        executor_type=resolved_executor_type,
        executor_kwargs=executor_kwargs,
        **kwargs,
    )

    # Attach the facade for callers who need direct access
    agent.facade = facade  # type: ignore[attr-defined]

    return agent
