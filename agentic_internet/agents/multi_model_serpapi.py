"""
SerpAPI Enhanced Context-Engineered AgenticInternet System
Multi-model orchestration with specialized model assignment for optimal performance
"""

import asyncio
import os
import json
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pydantic import BaseModel
from serpapi import GoogleSearch, GoogleScholarSearch, BingSearch, YahooSearch, BaiduSearch
from smolagents import (
    CodeAgent,
    ToolCallingAgent, 
    Tool,
    LiteLLMModel
)
from .basic_agent import BasicAgent

from dotenv import load_dotenv
load_dotenv()

# Multi-Model Configuration for Specialized Tasks
# Multi-Model Configuration for Specialized Tasks
class ModelManager:
    """Manages multiple LiteLLM models with strategic assignment"""
    
    def __init__(self):
        # Initialize comprehensive model suite
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Warning: OPENROUTER_API_KEY not found. Model initialization will be skipped.")
            self.models_available = False
            return
        
        self.models_available = True
        
        try:
            self.claude = LiteLLMModel(
                model_id="openrouter/anthropic/claude-sonnet-4.5", 
                api_key=api_key, 
                temperature=0.7
            )
        except Exception as e:
            print(f"Warning: Failed to initialize Claude model: {e}")
            self.claude = None
        
        # Initialize all other models with error handling
        models_config = [
            ('deepseek', "openrouter/deepseek/deepseek-r1-0528", 0.6),
            ('deepseek_chat', "openrouter/deepseek/deepseek-v3.2", 0.6),
            ('mistral_small', "openrouter/mistralai/mistral-small-3.2-24b-instruct", 0.8),
            ('sonar', "openrouter/perplexity/sonar", 0.8),
            ('sonar_reasoning', "openrouter/perplexity/sonar-reasoning", None),
            ('openai_o3', "openrouter/openai/o3-mini", None),
            ('mistral_large', "openrouter/mistralai/mistral-large-2411", 0.8),
            ('gemini', "openrouter/google/gemini-2.5-flash", 0.7),
            ('gpt_oss', "openrouter/openai/gpt-oss-120b", None),
            ('gemini_pro', "openrouter/google/gemini-2.5-pro", 0.6),
            ('chatgpt', "openrouter/openai/gpt-5-chat", 0.5),
            ('qwen3_coder', "openrouter/qwen3-coder", 0.6),
            ('qwen_big', "openrouter/qwen/qwen3-235b-a22b", 0.6),
            ('scout', "openrouter/meta-llama/llama-4-scout:nitro", 0.6),
            ('xai', "x-ai/grok-4", 0.8),
            ('gpt5', "openrouter/openai/gpt-5", 0.6),
            ('kimi', "openrouter/moonshotai/kimi-k2", 0.6)
        ]
        
        for attr_name, model_id, temp in models_config:
            try:
                if temp is not None:
                    model = LiteLLMModel(
                        model_id=model_id,
                        api_key=api_key,
                        temperature=temp
                    )
                else:
                    model = LiteLLMModel(
                        model_id=model_id,
                        api_key=api_key
                    )
                setattr(self, attr_name, model)
            except Exception as e:
                print(f"Warning: Failed to initialize {attr_name} model: {e}")
                setattr(self, attr_name, None)
    
    def get_model_for_role(self, role: str) -> Optional[LiteLLMModel]:
        """Strategic model assignment based on agent role and capabilities"""
        if not self.models_available:
            return None
        
        # First check if the role corresponds to a direct model name
        # This allows users to specify models like 'gpt-5', 'claude-4', etc.
        model_name_mappings = {
            'gpt-5': 'gpt5',
            'claude-4': 'claude',
            'claude': 'claude',
            'deepseek': 'deepseek',
            'gemini': 'gemini',
            'gemini-pro': 'gemini_pro',
            'sonar': 'sonar',
            'sonar-reasoning': 'sonar_reasoning',
            'mistral': 'mistral_large',
            'mistral-small': 'mistral_small',
            'mistral-large': 'mistral_large',
            'qwen': 'qwen_big',
            'qwen-coder': 'qwen3_coder',
            'xai': 'xai',
            'grok': 'xai',
            'kimi': 'kimi',
            'chatgpt': 'chatgpt',
            'o3': 'openai_o3'
        }
        
        # Check if the role is a direct model specification
        if role.lower() in model_name_mappings:
            model_attr = model_name_mappings[role.lower()]
            model = getattr(self, model_attr, None)
            if model is not None:
                print(f"Using specified model: {role}")
                return model
            else:
                print(f"Warning: Requested model {role} is not available")
        
        # Fall back to role-based assignments
        model_assignments = {
            # Orchestrator: Needs excellent reasoning and coordination
            "orchestrator": getattr(self, 'claude', None),  # Claude Sonnet 4 - Superior reasoning and planning
            
            # Search Researcher: Needs broad knowledge and search optimization
            "search_researcher": getattr(self, 'sonar', None),  # Perplexity Sonar - Optimized for search tasks
            
            # E-commerce Analyst: Needs structured analysis and price comparison
            "ecommerce_analyst": getattr(self, 'gemini_pro', None),  # Gemini 2.5 Pro - Excellent at structured data analysis
            
            # Local Business Analyst: Needs location awareness and business intelligence
            "local_business_analyst": getattr(self, 'mistral_large', None),  # Mistral Large - Good at business analysis
            
            # Academic Researcher: Needs research methodology and citation analysis
            "academic_researcher": getattr(self, 'sonar_reasoning', None),  # Perplexity Sonar Reasoning - Research-focused
            
            # Data Analyst: Needs coding and mathematical capabilities
            "data_analyst": getattr(self, 'qwen3_coder', None),  # Qwen3 Coder - Optimized for code and data analysis
            
            # Browser Navigator: Needs web interaction understanding
            "browser_navigator": getattr(self, 'gpt5', None),
            
            # Competitive Intelligence: Needs synthesis and strategic analysis
            "competitive_analyst": getattr(self, 'gpt5', None),  # GPT-5 - Advanced reasoning for strategic analysis
            
            # Market Research Synthesizer: Needs comprehensive analysis
            "market_synthesizer": getattr(self, 'qwen_big', None),  # Qwen3 235B - Massive context for synthesis
            
            # Creative/Content Tasks: Needs creative capabilities
            "content_creator": getattr(self, 'xai', None),  # Grok 3 - Creative and innovative thinking
            
            # Fast Response Tasks: Needs speed
            "quick_responder": getattr(self, 'gemini', None),  # Gemini 2.5 Flash - Speed optimized
            
            # Reasoning Tasks: Needs deep logical thinking
            "deep_reasoner": getattr(self, 'openai_o3', None),  # O3 Mini - Advanced reasoning
            
            # Conversational Tasks: Needs natural dialogue
            "conversationalist": getattr(self, 'chatgpt', None),  # GPT-5 Chat - Conversation optimized
            
            # Long Context Tasks: Needs extensive memory
            "long_context": getattr(self, 'kimi', None),  # Kimi K2 - Long context specialist
        }
        
        model = model_assignments.get(role, getattr(self, 'claude', None))  # Default to Claude
        
        # If the requested model is None, try to find any available model
        if model is None:
            for attr_name in ['claude', 'gemini', 'mistral_small', 'deepseek']:
                fallback_model = getattr(self, attr_name, None)
                if fallback_model is not None:
                    print(f"Warning: Using fallback model {attr_name} for role {role}")
                    return fallback_model
        
        return model
    
    def get_model_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed capabilities of each model for optimization"""
        return {
            "claude": {"strengths": ["reasoning", "planning", "analysis"], "best_for": "orchestration"},
            "deepseek": {"strengths": ["reasoning", "math", "logic"], "best_for": "complex_analysis"},
            "deepseek_chat": {"strengths": ["conversation", "reasoning"], "best_for": "interactive_tasks"},
            "mistral_small": {"strengths": ["efficiency", "speed"], "best_for": "quick_tasks"},
            "sonar": {"strengths": ["search", "web_knowledge"], "best_for": "research_tasks"},
            "sonar_reasoning": {"strengths": ["research", "citations"], "best_for": "academic_research"},
            "openai_o3": {"strengths": ["reasoning", "problem_solving"], "best_for": "complex_reasoning"},
            "mistral_large": {"strengths": ["analysis", "business"], "best_for": "business_intelligence"},
            "gemini": {"strengths": ["speed", "multimodal"], "best_for": "fast_response"},
            "gpt_oss": {"strengths": ["general", "balanced"], "best_for": "general_tasks"},
            "gemini_pro": {"strengths": ["analysis", "structured_data"], "best_for": "data_analysis"},
            "chatgpt": {"strengths": ["conversation", "creativity"], "best_for": "chat_tasks"},
            "qwen3_coder": {"strengths": ["coding", "technical"], "best_for": "programming_analysis"},
            "qwen_big": {"strengths": ["context", "synthesis"], "best_for": "comprehensive_analysis"},
            "scout": {"strengths": ["navigation", "exploration"], "best_for": "web_navigation"},
            "xai": {"strengths": ["creativity", "innovation"], "best_for": "creative_tasks"},
            "gpt5": {"strengths": ["advanced_reasoning", "strategy"], "best_for": "strategic_analysis"},
            "kimi": {"strengths": ["long_context", "memory"], "best_for": "extended_analysis"}
        }
# Enhanced Pydantic Models for SerpAPI
class SearchResult(BaseModel):
    position: Optional[int] = None
    title: str
    link: str
    snippet: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None

class LocalResult(BaseModel):
    title: str
    address: str
    phone: Optional[str] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None
    place_id: Optional[str] = None
    website: Optional[str] = None

class ShoppingResult(BaseModel):
    title: str
    price: str
    source: str
    link: str
    rating: Optional[float] = None
    reviews: Optional[int] = None
    shipping: Optional[str] = None

class NewsResult(BaseModel):
    title: str
    link: str
    source: str
    date: Optional[str] = None
    snippet: Optional[str] = None

class ScholarResult(BaseModel):
    title: str
    link: str
    authors: Optional[List[str]] = None
    publication: Optional[str] = None
    year: Optional[int] = None
    cited_by: Optional[int] = None
    snippet: Optional[str] = None

class ImageResult(BaseModel):
    title: str
    link: str
    original: str
    thumbnail: str
    source: str
    width: Optional[int] = None
    height: Optional[int] = None

# Context Engineering Components
@dataclass
class ContextWindow:
    """Manages context window size and content prioritization"""
    max_tokens: int = 8192
    current_tokens: int = 0
    priority_content: List[str] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    compression_threshold: float = 0.8
    
    def add_content(self, content: str, priority: int = 1) -> bool:
        estimated_tokens = len(content.split()) * 1.3
        
        if self.current_tokens + estimated_tokens > self.max_tokens * self.compression_threshold:
            self._compress_context()
        
        if priority > 3:
            self.priority_content.insert(0, content)
        else:
            self.priority_content.append(content)
        
        self.current_tokens += estimated_tokens
        return True
    
    def _compress_context(self):
        if len(self.priority_content) > 5:
            compressed_middle = f"[COMPRESSED: {len(self.priority_content[2:-2])} items summarized]"
            self.priority_content = (
                self.priority_content[:2] + 
                [compressed_middle] + 
                self.priority_content[-2:]
            )
            self.current_tokens = int(self.max_tokens * 0.6)

@dataclass
class AgentMemory:
    """Enhanced memory system with search pattern learning"""
    short_term: Dict[str, Any] = field(default_factory=dict)
    long_term: Dict[str, Any] = field(default_factory=dict)
    episodic: List[Dict[str, Any]] = field(default_factory=list)
    semantic: Dict[str, List[str]] = field(default_factory=dict)
    search_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_episodes: int = 50
    
    def remember_search_pattern(self, query_type: str, engine: str, success_metrics: Dict[str, Any]):
        """Remember successful search patterns for different query types"""
        pattern_key = f"{query_type}_{engine}"
        if pattern_key not in self.search_patterns:
            self.search_patterns[pattern_key] = {
                'success_count': 0,
                'total_attempts': 0,
                'avg_results': 0,
                'best_params': {},
                'common_issues': []
            }
        
        pattern = self.search_patterns[pattern_key]
        pattern['total_attempts'] += 1
        
        if success_metrics.get('success', False):
            pattern['success_count'] += 1
            pattern['avg_results'] = (pattern['avg_results'] + success_metrics.get('result_count', 0)) / 2
            if success_metrics.get('params'):
                pattern['best_params'] = success_metrics['params']

    def get_search_recommendations(self, query_type: str, engine: str) -> Dict[str, Any]:
        """Get recommendations based on past search patterns"""
        pattern_key = f"{query_type}_{engine}"
        if pattern_key in self.search_patterns:
            return self.search_patterns[pattern_key]
        return {}

@dataclass
class TaskContext:
    """Enhanced task context with search-specific tracking"""
    task_id: str
    objective: str
    current_step: int = 0
    total_steps: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)
    progress_markers: List[str] = field(default_factory=list)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    cross_engine_results: Dict[str, List[Dict]] = field(default_factory=dict)
    
    def log_search(self, engine: str, query: str, results_count: int, success: bool):
        """Log search attempts for analysis and optimization"""
        self.search_history.append({
            'engine': engine,
            'query': query,
            'results_count': results_count,
            'success': success,
            'timestamp': datetime.now(),
            'step': self.current_step
        })
    
    def get_progress_summary(self) -> str:
        """Get a summary of current task progress"""
        return f"Task: {self.task_id}, Step {self.current_step}/{self.total_steps}, Searches: {len(self.search_history)}"

# Advanced SerpAPI Tools
class GoogleSearchTool(Tool):
    name = "google_search"
    description = "Advanced Google search with location, language, and filter support using SerpAPI."
    inputs = {
        "query": {"type": "string", "description": "Search query"},
        "location": {"type": "string", "description": "Geographic location for search (optional)", "nullable": True},
        "language": {"type": "string", "description": "Language code (default: 'en')", "nullable": True},
        "country": {"type": "string", "description": "Country code (default: 'us')", "nullable": True},
        "num_results": {"type": "integer", "description": "Number of results (default: 10)", "nullable": True},
        "date_filter": {"type": "string", "description": "Date filter: qdr:d (day), qdr:w (week), qdr:m (month), qdr:y (year)", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine
    
    def forward(self, query: str, location: str = None, language: str = "en", 
                country: str = "us", num_results: int = 10, date_filter: str = None) -> str:
        """Execute Google search with advanced parameters"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."
        
        try:
            params = {
                'api_key': self.api_key,
                'q': query,
                'hl': language,
                'gl': country,
                'num': num_results
            }
            
            if location:
                params['location'] = location
            if date_filter:
                params['tbs'] = date_filter
            
            # Get recommendations from memory if context engine exists
            if self.context_engine:
                recommendations = self.context_engine.memory.get_search_recommendations('web_search', 'google')
                if recommendations.get('best_params'):
                    params.update(recommendations['best_params'])
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            # Process results
            organic_results = results.get('organic_results', [])
            processed_results = []
            
            for i, result in enumerate(organic_results):
                processed_results.append({
                    'position': i + 1,
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'source': result.get('source', '')
                })
            
            # Update context and memory if available
            if self.context_engine:
                success_metrics = {
                    'success': len(processed_results) > 0,
                    'result_count': len(processed_results),
                    'params': params
                }
                self.context_engine.memory.remember_search_pattern('web_search', 'google', success_metrics)
                
                if hasattr(self.context_engine, 'current_task_context') and self.context_engine.current_task_context:
                    self.context_engine.current_task_context.log_search('google', query, len(processed_results), True)
            
            return json.dumps({'results': processed_results, 'total': len(processed_results)}, indent=2)
            
        except Exception as e:
            error_msg = f"Google search failed: {str(e)}"
            if self.context_engine and hasattr(self.context_engine, 'current_task_context') and self.context_engine.current_task_context:
                self.context_engine.current_task_context.log_search('google', query, 0, False)
            return error_msg

class GoogleShoppingTool(Tool):
    name = "google_shopping"
    description = "Search Google Shopping for products with prices and reviews."
    inputs = {
        "query": {"type": "string", "description": "Product search query"},
        "location": {"type": "string", "description": "Shopping location", "nullable": True},
        "min_price": {"type": "integer", "description": "Minimum price filter", "nullable": True},
        "max_price": {"type": "integer", "description": "Maximum price filter", "nullable": True},
        "sort_by": {"type": "string", "description": "Sort by: r (rating), rv (reviews), p (price)", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine
    
    def forward(self, query: str, location: str = None, min_price: int = None, 
                max_price: int = None, sort_by: str = None) -> str:
        """Search Google Shopping with filters"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."
        
        try:
            params = {
                'api_key': self.api_key,
                'q': query
            }
            
            if location:
                params['location'] = location
            if min_price:
                params['min_price'] = min_price
            if max_price:
                params['max_price'] = max_price
            if sort_by:
                params['sort_by'] = sort_by
            
            # Google Shopping needs to be handled through GoogleSearch with shopping params
            search = GoogleSearch(params)
            search.params_dict['tbm'] = 'shop'
            results = search.get_dict()
            shopping_results = results.get('shopping_results', [])
            
            processed_results = []
            for result in shopping_results:
                processed_results.append({
                    'title': result.get('title', ''),
                    'price': result.get('price', ''),
                    'source': result.get('source', ''),
                    'link': result.get('link', ''),
                    'rating': result.get('rating'),
                    'reviews': result.get('reviews'),
                    'shipping': result.get('shipping')
                })
            
            # Update memory if available
            if self.context_engine:
                success_metrics = {
                    'success': len(processed_results) > 0,
                    'result_count': len(processed_results),
                    'params': params
                }
                self.context_engine.memory.remember_search_pattern('shopping', 'google', success_metrics)
            
            return json.dumps({'shopping_results': processed_results, 'total': len(processed_results)}, indent=2)
            
        except Exception as e:
            return f"Google Shopping search failed: {str(e)}"

class GoogleMapsLocalTool(Tool):
    name = "google_maps_local"
    description = "Search Google Maps for local businesses and places."
    inputs = {
        "query": {"type": "string", "description": "Local business search query"},
        "location": {"type": "string", "description": "Search location (required)"},
        "type": {"type": "string", "description": "Place type filter (restaurant, hotel, etc.)", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine
    
    def forward(self, query: str, location: str, type: str = None) -> str:
        """Search Google Maps for local businesses"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."
        
        try:
            params = {
                'api_key': self.api_key,
                'q': query,
                'll': f'@{location}' if not ',' in location else location,
                'type': 'search'
            }
            
            if type:
                params['type'] = type
            
            # Google Maps search through regular Google Search
            search = GoogleSearch(params)
            results = search.get_dict()
            local_results = results.get('local_results', [])
            
            processed_results = []
            for result in local_results:
                processed_results.append({
                    'title': result.get('title', ''),
                    'address': result.get('address', ''),
                    'phone': result.get('phone'),
                    'rating': result.get('rating'),
                    'reviews': result.get('reviews'),
                    'website': result.get('website'),
                    'place_id': result.get('place_id')
                })
            
            return json.dumps({'local_results': processed_results, 'total': len(processed_results)}, indent=2)
            
        except Exception as e:
            return f"Google Maps search failed: {str(e)}"

class GoogleScholarTool(Tool):
    name = "google_scholar"
    description = "Search Google Scholar for academic papers and citations."
    inputs = {
        "query": {"type": "string", "description": "Academic search query"},
        "year_low": {"type": "integer", "description": "Start year filter", "nullable": True},
        "year_high": {"type": "integer", "description": "End year filter", "nullable": True},
        "sort_by": {"type": "string", "description": "Sort by: relevance or date", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine
    
    def forward(self, query: str, year_low: int = None, year_high: int = None, sort_by: str = None) -> str:
        """Search Google Scholar for academic content"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."
        
        try:
            params = {
                'api_key': self.api_key,
                'q': query
            }
            
            if year_low:
                params['as_ylo'] = year_low
            if year_high:
                params['as_yhi'] = year_high
            if sort_by == 'date':
                params['scisbd'] = 1
            
            # Use GoogleScholarSearch for Scholar queries
            search = GoogleScholarSearch(params)
            results = search.get_dict()
            organic_results = results.get('organic_results', [])
            
            processed_results = []
            for result in organic_results:
                processed_results.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'publication_info': result.get('publication_info', {})
                })
            
            return json.dumps({'scholar_results': processed_results, 'total': len(processed_results)}, indent=2)
            
        except Exception as e:
            return f"Google Scholar search failed: {str(e)}"

class MultiEngineSearchTool(Tool):
    name = "multi_engine_search"
    description = "Search across multiple search engines (Google, Bing, Yahoo, Baidu) and compare results."
    inputs = {
        "query": {"type": "string", "description": "Search query"},
        "engines": {"type": "string", "description": "Comma-separated engines (google,bing,yahoo,baidu)", "nullable": True},
        "compare": {"type": "boolean", "description": "Whether to compare results across engines", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, api_key: str, context_engine=None):
        super().__init__()
        self.api_key = api_key
        self.context_engine = context_engine
    
    def forward(self, query: str, engines: str = "google,bing", compare: bool = True) -> str:
        """Search multiple engines and optionally compare results"""
        if not self.api_key:
            return "SerpAPI client not initialized. Please provide an API key."
        
        try:
            engine_list = [e.strip() for e in engines.split(',')]
            all_results = {}
            
            for engine in engine_list:
                try:
                    params = {'api_key': self.api_key, 'q': query}
                    
                    # Use appropriate search class for each engine
                    if engine == 'google':
                        search = GoogleSearch(params)
                    elif engine == 'bing':
                        search = BingSearch(params)
                    elif engine == 'yahoo':
                        search = YahooSearch(params)
                    elif engine == 'baidu':
                        search = BaiduSearch(params)
                    else:
                        all_results[engine] = f"Error: Unsupported engine {engine}"
                        continue
                    
                    results = search.get_dict()
                    
                    organic_results = results.get('organic_results', [])
                    processed = [
                        {
                            'position': i + 1,
                            'title': r.get('title', ''),
                            'link': r.get('link', ''),
                            'snippet': r.get('snippet', '')
                        }
                        for i, r in enumerate(organic_results[:10])
                    ]
                    
                    all_results[engine] = processed
                    
                    # Store results for cross-engine comparison
                    if self.context_engine and hasattr(self.context_engine, 'current_task_context') and self.context_engine.current_task_context:
                        if engine not in self.context_engine.current_task_context.cross_engine_results:
                            self.context_engine.current_task_context.cross_engine_results[engine] = []
                        self.context_engine.current_task_context.cross_engine_results[engine].extend(processed)
                    
                except Exception as e:
                    all_results[engine] = f"Error: {str(e)}"
            
            if compare and len(all_results) > 1:
                # Find common results across engines
                common_links = self._find_common_results(all_results)
                all_results['comparison'] = {
                    'common_results': common_links,
                    'engine_coverage': {engine: len(results) if isinstance(results, list) else 0 
                                      for engine, results in all_results.items() if engine != 'comparison'}
                }
            
            return json.dumps(all_results, indent=2)
            
        except Exception as e:
            return f"Multi-engine search failed: {str(e)}"
    
    def _find_common_results(self, all_results: Dict) -> List[Dict]:
        """Find results that appear across multiple engines"""
        common_results = []
        engine_results = {k: v for k, v in all_results.items() if isinstance(v, list)}
        
        if len(engine_results) < 2:
            return []
        
        # Get first engine results as baseline
        baseline_engine = list(engine_results.keys())[0]
        baseline_results = engine_results[baseline_engine]
        
        for result in baseline_results:
            result_link = result.get('link', '')
            if not result_link:
                continue
                
            # Check if this link appears in other engines
            appearances = [baseline_engine]
            for other_engine, other_results in engine_results.items():
                if other_engine == baseline_engine:
                    continue
                    
                for other_result in other_results:
                    if other_result.get('link', '') == result_link:
                        appearances.append(other_engine)
                        break
            
            if len(appearances) > 1:
                common_results.append({
                    **result,
                    'appears_in': appearances,
                    'cross_engine_rank': len(appearances)
                })
        
        return sorted(common_results, key=lambda x: x['cross_engine_rank'], reverse=True)

class AgentTool(Tool):
    """Wrapper to use agents as tools for orchestration"""
    
    def __init__(self, agent, agent_name: str, agent_description: str):
        self.name = agent_name
        self.description = agent_description
        self.inputs = {'task': {'type': 'string', 'description': 'Task to delegate to this specialized agent'}}
        self.output_type = 'string'
        super().__init__()
        self.agent = agent
    
    def forward(self, task: str) -> str:
        """Execute the agent with the given task"""
        try:
            if self.agent:
                result = self.agent.run(task)
                return str(result)
            return f"Agent {self.name} is not available"
        except Exception as e:
            return f"Agent {self.name} failed: {str(e)}"

class ContextEngineeringMixin:
    """Enhanced mixin with SerpAPI context awareness"""
    
    def __init__(self):
        self.context_window = ContextWindow()
        self.memory = AgentMemory()
        self.current_task_context: Optional[TaskContext] = None
    
    def prepare_search_context(self, query: str, search_type: str, engine: str) -> Dict[str, Any]:
        """Prepare context-aware search parameters based on query and history"""
        context = {
            'base_query': query,
            'search_type': search_type,
            'engine': engine,
            'recommendations': self.memory.get_search_recommendations(search_type, engine)
        }
        
        # Add relevant search history
        if self.current_task_context:
            relevant_searches = [
                s for s in self.current_task_context.search_history 
                if s['engine'] == engine and s['success']
            ]
            context['recent_successful_searches'] = relevant_searches[-3:]  # Last 3 successful
        
        return context
    
    def prepare_agent_context(self, task: str, agent_role: str) -> str:
        """Prepare context for agent initialization"""
        return f"""
Task: {task}
Role: {agent_role}
Memory Status: {len(self.memory.episodic)} episodes, {len(self.memory.search_patterns)} search patterns
Current Context: {self.context_window.current_tokens}/{self.context_window.max_tokens} tokens
"""
    
    def update_context_from_result(self, agent_name: str, task: str, result: Any, success: bool):
        """Update context based on agent results"""
        self.memory.episodic.append({
            'agent': agent_name,
            'task': task,
            'result': str(result)[:500],  # Truncate for memory
            'success': success,
            'timestamp': datetime.now()
        })
        
        # Keep episodic memory bounded
        if len(self.memory.episodic) > self.memory.max_episodes:
            self.memory.episodic = self.memory.episodic[-self.memory.max_episodes:]
    
    def analyze_search_performance(self) -> Dict[str, Any]:
        """Analyze search performance across engines and query types"""
        if not self.current_task_context or not self.current_task_context.search_history:
            return {'message': 'No search history available'}
        
        analysis = {
            'total_searches': len(self.current_task_context.search_history),
            'success_rate': 0,
            'engine_performance': {},
            'query_patterns': {},
            'recommendations': []
        }
        
        successful_searches = [s for s in self.current_task_context.search_history if s['success']]
        analysis['success_rate'] = len(successful_searches) / len(self.current_task_context.search_history)
        
        # Analyze by engine
        for search in self.current_task_context.search_history:
            engine = search['engine']
            if engine not in analysis['engine_performance']:
                analysis['engine_performance'][engine] = {
                    'total': 0, 'successful': 0, 'avg_results': 0
                }
            
            perf = analysis['engine_performance'][engine]
            perf['total'] += 1
            if search['success']:
                perf['successful'] += 1
                perf['avg_results'] = (perf['avg_results'] + search['results_count']) / perf['successful']
        
        # Calculate success rates
        for engine, perf in analysis['engine_performance'].items():
            perf['success_rate'] = perf['successful'] / perf['total'] if perf['total'] > 0 else 0
        
        return analysis
    
    def _analyze_cross_engine_results(self) -> Dict[str, Any]:
        """Analyze results across different search engines"""
        if not self.current_task_context or not self.current_task_context.cross_engine_results:
            return {'message': 'No cross-engine results available'}
        
        analysis = {
            'engines_used': list(self.current_task_context.cross_engine_results.keys()),
            'total_unique_results': 0,
            'common_results': 0,
            'engine_specific_results': {}
        }
        
        # Find unique results across all engines
        all_links = set()
        for engine, results in self.current_task_context.cross_engine_results.items():
            engine_links = {r.get('link', '') for r in results if isinstance(r, dict)}
            analysis['engine_specific_results'][engine] = len(engine_links)
            all_links.update(engine_links)
        
        analysis['total_unique_results'] = len(all_links)
        
        return analysis
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of system state and performance"""
        return {
            'memory_state': {
                'short_term_items': len(self.memory.short_term),
                'long_term_items': len(self.memory.long_term),
                'episodic_memories': len(self.memory.episodic),
                'search_patterns': len(self.memory.search_patterns)
            },
            'context_state': {
                'current_tokens': self.context_window.current_tokens,
                'max_tokens': self.context_window.max_tokens,
                'priority_items': len(self.context_window.priority_content)
            },
            'task_state': {
                'current_task': self.current_task_context.task_id if self.current_task_context else None,
                'progress': f"{self.current_task_context.current_step}/{self.current_task_context.total_steps}" if self.current_task_context else "N/A",
                'search_history_count': len(self.current_task_context.search_history) if self.current_task_context else 0
            },
            'performance_analysis': self.analyze_search_performance() if self.current_task_context else {}
        }

class MultiModelSerpAPISystem(ContextEngineeringMixin):
    """
    Enhanced orchestrator with strategic multi-model assignment and SerpAPI integration
    """

    def __init__(self, 
                 serpapi_key: str = None,
                 context_window_size: int = 16384):
        # Initialize context engineering
        super().__init__()
        self.context_window = ContextWindow(max_tokens=context_window_size)
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize SerpAPI
        self.serpapi_key = serpapi_key or os.environ.get("SERPAPI_API_KEY")
        
        self.workers: Dict[str, Union[CodeAgent, ToolCallingAgent]] = {}
        self.worker_tools: Dict[str, AgentTool] = {}  # Store wrapped agents as tools
        self.model_performance_tracker = {}

    def create_serpapi_tools(self) -> List[Tool]:
        """Create comprehensive SerpAPI tool suite"""
        if not self.serpapi_key:
            print("Warning: No SerpAPI key provided. Search tools will be limited.")
            return []
        
        return [
            GoogleSearchTool(self.serpapi_key, self),
            GoogleShoppingTool(self.serpapi_key, self),
            GoogleMapsLocalTool(self.serpapi_key, self),
            GoogleScholarTool(self.serpapi_key, self),
            MultiEngineSearchTool(self.serpapi_key, self)
        ]
    
    def _check_model_tool_support(self, model) -> bool:
        """Check if a model supports tool calling"""
        # Models known not to support tool calling through OpenRouter
        # IMPORTANT: Perplexity models technically support tools via LiteLLM,
        # but OpenRouter cannot route them with tool use (returns 404 error)
        non_tool_models = [
            'perplexity/sonar',
            'perplexity/sonar-reasoning',
            'perplexity/llama',  # Any Perplexity models
        ]
        
        # Check if model has a model_id attribute and if it contains non-tool model names
        if hasattr(model, 'model_id'):
            model_id = str(model.model_id).lower()
            for non_tool_model in non_tool_models:
                if non_tool_model in model_id:
                    return False
        
        # By default, assume the model supports tools
        # Most models through LiteLLM/OpenRouter support tool calling
        return True
    
    def _check_model_code_support(self, model) -> bool:
        """Check if a model supports code generation for CodeAgent"""
        # Models that are good at code generation
        code_capable_models = [
            'claude', 'gpt', 'deepseek', 'mistral', 'gemini', 'qwen'
        ]
        
        if hasattr(model, 'model_id'):
            model_id = str(model.model_id).lower()
            for code_model in code_capable_models:
                if code_model in model_id:
                    return True
        
        # Models like Perplexity are optimized for search/QA, not code generation
        return False

    def create_specialized_worker(self, name: str, description: str, 
                                 tools: List[Tool], agent_type: str = "ToolCallingAgent",
                                 model_override: str = None) -> None:
        """Create worker with optimal model selection and enhanced capabilities"""
        
        # Get optimal model for this role
        model = self.model_manager.get_model_for_role(model_override or name)
        
        if model is None:
            print(f"Warning: No model available for {name}. Skipping worker creation.")
            return
        
        model_info = self.model_manager.get_model_capabilities()
        model_name = model_override or name
        capabilities = model_info.get(model_name, {})
        
        # Enhanced context with model-specific optimization
        model_context = f"""
SPECIALIZED MODEL ASSIGNMENT:
Model strengths: {', '.join(capabilities.get('strengths', ['general']))}

SERPAPI MULTI-ENGINE CAPABILITIES:
- Google Web Search: Comprehensive with location/language/date filters
- Google Shopping: Product search with price/review filtering  
- Google Maps/Local: Business search with ratings/reviews
- Google Scholar: Academic papers and citation analysis
- Multi-Engine Search: Cross-platform comparison and validation

CONTEXT-AWARE OPTIMIZATION:
1. Learn from previous successful search patterns
2. Adapt search parameters based on query type and past performance
3. Use cross-engine validation for critical information
4. Track and optimize search performance metrics
5. Synthesize results with model-specific analytical strengths
"""
        
        base_context = self.prepare_agent_context("initialization", name)
        enhanced_description = f"""
{description}

{model_context}

CONTEXT AWARENESS:
{base_context}

MODEL-OPTIMIZED APPROACH:
- Leverage your specific model strengths: {', '.join(capabilities.get('strengths', []))}
- Focus on tasks you excel at: {capabilities.get('best_for', 'general tasks')}
- Coordinate with other specialized agents for comprehensive results
- Provide analysis depth appropriate to your model capabilities
"""
        
        try:
            # Create prompt templates with the enhanced description as system prompt
            from smolagents.agents import (
                PromptTemplates, PlanningPromptTemplate, 
                ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
            )
            
            # Create empty sub-templates with required keys
            planning_template = PlanningPromptTemplate(
                initial_plan=None,
                update_plan_pre_messages=None,
                update_plan_post_messages=None
            )
            
            managed_template = ManagedAgentPromptTemplate(
                task=None,
                report=None
            )
            
            final_template = FinalAnswerPromptTemplate(
                pre_messages=None,
                post_messages=None
            )
            
            # Create full prompt templates
            prompt_templates = PromptTemplates(
                system_prompt=enhanced_description,
                planning=planning_template,
                managed_agent=managed_template,
                final_answer=final_template
            )
            
            # Check model capabilities
            model_supports_tools = self._check_model_tool_support(model)
            model_supports_code = self._check_model_code_support(model)
            
            if agent_type == "CodeAgent" and model_supports_code:
                # Use CodeAgent for models that support code generation
                agent = CodeAgent(
                    tools=tools,
                    model=model,
                    prompt_templates=prompt_templates,
                    max_steps=20,
                    additional_authorized_imports=[
                        "pandas", "numpy", "json", "csv", "re", "datetime", 
                        "time", "requests", "urllib", "math", "statistics"
                    ]
                )
            elif model_supports_tools:
                # Use ToolCallingAgent for models that support tool calling
                agent = ToolCallingAgent(
                    tools=tools,
                    model=model,
                    prompt_templates=prompt_templates
                )
            else:
                # Use BasicAgent for models that don't support tools or code
                agent = BasicAgent(
                    model=model,
                    tools=tools,
                    prompt_templates=prompt_templates
                )
                print(f"Note: Model for {name} doesn't support tools or code generation, using BasicAgent")

            # Store both the raw agent and create an AgentTool wrapper
            self.workers[name] = agent
            
            # Create AgentTool wrapper for orchestrator use
            agent_tool = AgentTool(
                agent=agent,
                agent_name=name,
                agent_description=description
            )
            self.worker_tools[name] = agent_tool
            
        except Exception as e:
            print(f"Failed to create specialized worker {name}: {str(e)}")

    def setup_multi_model_workers(self, default_model: str = None):
        """Setup workers with strategic model assignments
        
        Args:
            default_model: Model to use for all workers (if not specified, uses role-based assignment)
        """
        
        # Create specialized tool sets
        serpapi_tools = self.create_serpapi_tools()
        
        if not serpapi_tools:
            print("Warning: No SerpAPI tools available. System will have limited functionality.")
            return
        
        # 🧠 Strategic Research Orchestrator
        research_tools = [t for t in serpapi_tools if t.name in ['google_search', 'google_scholar', 'multi_engine_search']]
        if research_tools:
            self.create_specialized_worker(
                name="search_researcher",
                description="Strategic research specialist with multi-engine search and cross-validation capabilities.",
                tools=research_tools,
                agent_type="ToolCallingAgent",
                model_override=default_model  # Use the passed model if available
            )

        # 💰 E-commerce Intelligence Specialist
        ecommerce_tools = [t for t in serpapi_tools if t.name in ['google_shopping', 'google_search']]
        if ecommerce_tools:
            self.create_specialized_worker(
                name="ecommerce_analyst",
                description="E-commerce and market analysis specialist with pricing intelligence.",
                tools=ecommerce_tools,
                agent_type="ToolCallingAgent",
                model_override=default_model  # Use the passed model if available
            )

        # 📍 Local Business Intelligence
        local_tools = [t for t in serpapi_tools if t.name in ['google_maps_local', 'google_search']]
        if local_tools:
            self.create_specialized_worker(
                name="local_business_analyst",
                description="Local market and business intelligence specialist.",
                tools=local_tools,
                agent_type="ToolCallingAgent",
                model_override=default_model  # Use the passed model if available
            )

        # 📚 Academic Research Specialist
        academic_tools = [t for t in serpapi_tools if t.name in ['google_scholar', 'google_search']]
        if academic_tools:
            self.create_specialized_worker(
                name="academic_researcher",
                description="Academic research and citation analysis specialist.",
                tools=academic_tools,
                agent_type="ToolCallingAgent",
                model_override=default_model  # Use the passed model if available
            )

    async def execute_multi_model_workflow(self, task: str, timeout: Optional[float] = None,
                                          orchestrator_model: str = "orchestrator") -> str:
        """Execute workflow with multi-model coordination and optimization"""
        
        # Initialize enhanced task context
        task_id = f"multimodel_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_task_context = TaskContext(
            task_id=task_id,
            objective=task,
            total_steps=10
        )
        
        if not self.workers:
            # Use the same model for all workers as the orchestrator for consistency
            self.setup_multi_model_workers(default_model=orchestrator_model)

        if not self.workers:
            return json.dumps({
                'error': 'No workers available. Please check API keys and configuration.',
                'task': task
            })

        try:
            # Get optimal orchestrator model
            orchestrator_model_instance = self.model_manager.get_model_for_role(orchestrator_model)
            
            if orchestrator_model_instance is None:
                return json.dumps({
                    'error': 'No orchestrator model available. Please check API keys.',
                    'task': task
                })
            
            # Prepare multi-model orchestrator context
            orchestrator_context = self.prepare_agent_context(task, "orchestrator")
            
            available_specialists = "\n".join([
                f"- {name}: specialized agent"
                for name in self.workers.keys()
            ])
            
            # Enhanced orchestrator with multi-model awareness
            orchestrator_prompt = f"""
You are a CODE-BASED ORCHESTRATOR managing a team of AI specialist agents through Python code.

{orchestrator_context}

AVAILABLE SPECIALIST AGENT TOOLS (call them as functions):
{available_specialists}

HOW TO USE THE AGENTS:
Each specialist agent is available as a tool that can be called like a function:
- academic_researcher(task="research query") - for historical/academic research
- search_researcher(task="search query") - for general web research  
- ecommerce_analyst(task="product/market query") - for commerce analysis
- local_business_analyst(task="business query") - for local business research

IMPORTANT INSTRUCTIONS:
1. You MUST write Python code to call these agent tools
2. Once you have obtained a satisfactory answer to the user's question, call final_answer() to complete
3. DO NOT continue generating completion messages after calling final_answer()
4. Use this exact code pattern to signal completion:
   ```python
   final_answer("Your complete answer here")
   ```

Example:
```python
result = academic_researcher(task="Who was the biggest merchant in medieval Mali")
print(result)
final_answer("Mansa Musa was the biggest merchant in medieval Mali, controlling vast gold and salt trade networks")
```

CURRENT TASK CONTEXT:
{self.current_task_context.get_progress_summary()}

Write Python code to execute the task. Remember to STOP after providing the final answer.
"""
            
            # Create multi-model orchestrator with both SerpAPI tools and wrapped agent tools
            orchestrator_tools = self.create_serpapi_tools()[:2] if self.serpapi_key else []
            
            # Add the wrapped worker agents as tools for the orchestrator
            orchestrator_tools.extend(list(self.worker_tools.values()))
            
            # Create prompt templates for orchestrator
            from smolagents.agents import (
                PromptTemplates, PlanningPromptTemplate,
                ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
            )
            
            # Create sub-templates for orchestrator
            orchestrator_planning = PlanningPromptTemplate(
                initial_plan=None,
                update_plan_pre_messages=None,
                update_plan_post_messages=None
            )
            
            orchestrator_managed = ManagedAgentPromptTemplate(
                task=None,
                report=None
            )
            
            orchestrator_final = FinalAnswerPromptTemplate(
                pre_messages=None,
                post_messages=None
            )
            
            orchestrator_templates = PromptTemplates(
                system_prompt=orchestrator_prompt,
                planning=orchestrator_planning,
                managed_agent=orchestrator_managed,
                final_answer=orchestrator_final
            )
            
            orchestrator = CodeAgent(
                tools=orchestrator_tools,
                model=orchestrator_model_instance,
                prompt_templates=orchestrator_templates,
                additional_authorized_imports=[
                    "pandas", "numpy", "json", "csv", "re", "datetime", 
                    "time", "requests", "urllib", "math", "statistics"
                ],
                max_steps=10  # Reduced from 50 to prevent infinite loops
            )

            # Execute with comprehensive tracking
            if timeout:
                result = await asyncio.wait_for(
                    asyncio.to_thread(orchestrator.run, task),
                    timeout=timeout
                )
            else:
                result = await asyncio.to_thread(orchestrator.run, task)

            # Enhanced result processing
            final_result = {
                'primary_result': result,
                'search_performance': self.analyze_search_performance(),
                'cross_engine_analysis': self._analyze_cross_engine_results()
            }
            
            # Update memory with results
            self.update_context_from_result("multi_model_orchestrator", task, result, True)
            
            return json.dumps(final_result, indent=2, default=str)

        except asyncio.TimeoutError:
            error_msg = f"Multi-model workflow execution exceeded {timeout} seconds"
            self.update_context_from_result("multi_model_orchestrator", task, error_msg, False)
            return json.dumps({'error': error_msg, 'task': task})
        except Exception as e:
            error_msg = f"Multi-model workflow execution failed: {str(e)}"
            self.update_context_from_result("multi_model_orchestrator", task, error_msg, False)
            return json.dumps({'error': error_msg, 'task': task})
