#!/usr/bin/env python
"""Test script to directly test an agent with GPT-5 and SerpAPI tools"""

import os
from smolagents import ToolCallingAgent, LiteLLMModel
from agentic_internet.agents.multi_model_serpapi import GoogleSearchTool

# Initialize model
api_key = os.environ.get("OPENROUTER_API_KEY")
serpapi_key = os.environ.get("SERPAPI_API_KEY")

if not api_key:
    print("Error: OPENROUTER_API_KEY not set")
    exit(1)

if not serpapi_key:
    print("Error: SERPAPI_API_KEY not set")
    exit(1)

# Create GPT-5 model
model = LiteLLMModel(
    model_id="openrouter/openai/gpt-5",
    api_key=api_key,
    temperature=0.7
)

print(f"Model initialized: {model.model_id}")

# Create search tool
search_tool = GoogleSearchTool(serpapi_key)

# Create agent
agent = ToolCallingAgent(
    tools=[search_tool],
    model=model
)

print("Agent created with GPT-5 and GoogleSearchTool")

# Test the agent
task = "Search for: What are the top 3 programming languages in 2024 according to TIOBE index?"
print(f"\nTask: {task}")
print("Running agent...")

try:
    result = agent.run(task)
    print(f"\nResult:\n{result}")
except Exception as e:
    print(f"\nError: {e}")
