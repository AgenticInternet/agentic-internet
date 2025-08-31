"""Specialized agent implementations for different use cases."""

from typing import Optional, List, Dict, Any
from .internet_agent import InternetAgent
from smolagents import Tool
from rich.console import Console
import json

console = Console()


class BrowserAutomationAgent(InternetAgent):
    """
    Specialized agent for browser automation tasks.
    Uses Browser Use tools for complex web interactions.
    """
    
    def __init__(self, **kwargs):
        """Initialize browser automation agent with browser-specific configuration."""
        # Default to code agent for better browser interaction capabilities
        kwargs.setdefault('agent_type', 'code')
        kwargs.setdefault('max_iterations', 20)
        super().__init__(**kwargs)
    
    def scrape_structured_data(self, url: str, data_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scrape structured data from a website.
        
        Args:
            url: The URL to scrape
            data_schema: Schema describing the expected data structure
            
        Returns:
            Extracted data matching the schema
        """
        task = f"""
        Navigate to {url} and extract data according to this schema:
        {json.dumps(data_schema, indent=2)}
        
        Return the extracted data in JSON format.
        """
        
        result = self.run(task)
        
        # Try to parse as JSON
        try:
            return json.loads(result)
        except:
            return {"raw_result": result}
    
    def fill_form(self, url: str, form_data: Dict[str, str]) -> str:
        """
        Fill and submit a form on a website.
        
        Args:
            url: The URL containing the form
            form_data: Dictionary of field names/IDs and values
            
        Returns:
            Result of form submission
        """
        task = f"""
        Navigate to {url} and fill the form with this data:
        {json.dumps(form_data, indent=2)}
        
        Submit the form and return the result or confirmation message.
        """
        
        return self.run(task)
    
    def monitor_website(self, url: str, check_for: str, interval_minutes: int = 30) -> Dict[str, Any]:
        """
        Monitor a website for specific changes or content.
        
        Args:
            url: The URL to monitor
            check_for: What to look for (changes, specific content, etc.)
            interval_minutes: How often to check (for scheduling purposes)
            
        Returns:
            Monitoring result
        """
        task = f"""
        Check the website at {url} for: {check_for}
        
        Extract relevant information and note any changes or findings.
        """
        
        result = self.run(task)
        
        return {
            "url": url,
            "check_for": check_for,
            "findings": result,
            "suggested_interval": f"{interval_minutes} minutes"
        }


class DataAnalysisAgent(InternetAgent):
    """
    Specialized agent for data analysis and visualization tasks.
    Optimized for working with structured data and generating insights.
    """
    
    def __init__(self, **kwargs):
        """Initialize data analysis agent with analysis-specific configuration."""
        # Use code agent for better data manipulation capabilities
        kwargs.setdefault('agent_type', 'code')
        # Only add imports that are actually installed
        # You can install matplotlib, seaborn, plotly with: uv add matplotlib seaborn plotly
        kwargs.setdefault('additional_authorized_imports', [])
        super().__init__(**kwargs)
    
    def analyze_dataset(self, data: Any, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze a dataset and provide insights.
        
        Args:
            data: The data to analyze (can be list, dict, or pandas DataFrame)
            analysis_type: Type of analysis ("basic", "comprehensive", "statistical")
            
        Returns:
            Analysis results with insights
        """
        task = f"""
        Analyze this dataset:
        {data}
        
        Perform a {analysis_type} analysis including:
        - Summary statistics
        - Key patterns and trends
        - Anomalies or outliers
        - Visualizations if appropriate
        - Actionable insights
        """
        
        result = self.run(task)
        
        return {
            "analysis_type": analysis_type,
            "insights": result
        }
    
    def compare_datasets(self, dataset1: Any, dataset2: Any, comparison_criteria: List[str] = None) -> str:
        """
        Compare two datasets and highlight differences.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            comparison_criteria: Specific aspects to compare
            
        Returns:
            Comparison results
        """
        criteria_str = "\n".join(f"- {c}" for c in comparison_criteria) if comparison_criteria else "all aspects"
        
        task = f"""
        Compare these two datasets:
        
        Dataset 1: {dataset1}
        Dataset 2: {dataset2}
        
        Focus on comparing: {criteria_str}
        
        Provide a detailed comparison with visualizations if helpful.
        """
        
        return self.run(task)


class ContentCreationAgent(InternetAgent):
    """
    Specialized agent for content creation and writing tasks.
    Optimized for generating high-quality written content.
    """
    
    def __init__(self, **kwargs):
        """Initialize content creation agent."""
        kwargs.setdefault('agent_type', 'tool_calling')
        super().__init__(**kwargs)
    
    def write_article(self, topic: str, style: str = "informative", 
                     word_count: int = 500, sources_required: bool = True) -> Dict[str, str]:
        """
        Write an article on a given topic.
        
        Args:
            topic: The topic to write about
            style: Writing style (informative, persuasive, technical, casual)
            word_count: Target word count
            sources_required: Whether to include sources/citations
            
        Returns:
            Article with metadata
        """
        sources_instruction = "Include citations and sources." if sources_required else ""
        
        task = f"""
        Write a {style} article about {topic}.
        Target length: approximately {word_count} words.
        {sources_instruction}
        
        Research the topic first, then write a well-structured article with:
        - Engaging introduction
        - Clear main points
        - Supporting evidence
        - Conclusion
        """
        
        result = self.run(task)
        
        return {
            "topic": topic,
            "style": style,
            "content": result,
            "word_count_target": word_count
        }
    
    def summarize_content(self, content: str, summary_type: str = "executive") -> str:
        """
        Summarize content in various formats.
        
        Args:
            content: Content to summarize (can be URL or text)
            summary_type: Type of summary (executive, bullet_points, abstract, tldr)
            
        Returns:
            Summarized content
        """
        task = f"""
        Summarize the following content as a {summary_type} summary:
        
        {content}
        
        Make it concise yet comprehensive, capturing all key points.
        """
        
        return self.run(task)


class MarketResearchAgent(InternetAgent):
    """
    Specialized agent for market research and competitive analysis.
    """
    
    def __init__(self, **kwargs):
        """Initialize market research agent."""
        kwargs.setdefault('agent_type', 'tool_calling')
        kwargs.setdefault('max_iterations', 15)
        super().__init__(**kwargs)
    
    def analyze_competitor(self, company_name: str, aspects: List[str] = None) -> Dict[str, Any]:
        """
        Perform competitive analysis on a company.
        
        Args:
            company_name: Name of the company to analyze
            aspects: Specific aspects to analyze (products, pricing, marketing, etc.)
            
        Returns:
            Competitive analysis report
        """
        aspects_list = aspects or ["products", "pricing", "market_position", "strengths", "weaknesses"]
        aspects_str = ", ".join(aspects_list)
        
        task = f"""
        Perform a comprehensive competitive analysis of {company_name}.
        
        Focus on these aspects: {aspects_str}
        
        Include:
        - Company overview
        - Market position
        - Key products/services
        - Pricing strategy
        - Marketing approach
        - SWOT analysis
        - Recent news and developments
        """
        
        result = self.run(task)
        
        return {
            "company": company_name,
            "aspects_analyzed": aspects_list,
            "analysis": result
        }
    
    def market_trends(self, industry: str, timeframe: str = "current") -> Dict[str, Any]:
        """
        Analyze market trends in a specific industry.
        
        Args:
            industry: Industry to analyze
            timeframe: Timeframe for trends (current, 2024, last_quarter, etc.)
            
        Returns:
            Market trends analysis
        """
        task = f"""
        Analyze {timeframe} market trends in the {industry} industry.
        
        Include:
        - Key trends and drivers
        - Market size and growth
        - Major players
        - Emerging technologies/innovations
        - Challenges and opportunities
        - Future outlook
        
        Provide data-backed insights with sources.
        """
        
        result = self.run(task)
        
        return {
            "industry": industry,
            "timeframe": timeframe,
            "trends": result
        }


class TechnicalSupportAgent(InternetAgent):
    """
    Specialized agent for technical support and troubleshooting.
    """
    
    def __init__(self, **kwargs):
        """Initialize technical support agent."""
        kwargs.setdefault('agent_type', 'code')
        super().__init__(**kwargs)
    
    def troubleshoot(self, problem_description: str, system_info: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Troubleshoot a technical problem.
        
        Args:
            problem_description: Description of the problem
            system_info: Optional system information (OS, software versions, etc.)
            
        Returns:
            Troubleshooting steps and solutions
        """
        system_context = f"System info: {json.dumps(system_info, indent=2)}" if system_info else ""
        
        task = f"""
        Troubleshoot this technical problem:
        {problem_description}
        
        {system_context}
        
        Provide:
        1. Likely causes
        2. Step-by-step troubleshooting guide
        3. Potential solutions
        4. Preventive measures
        """
        
        result = self.run(task)
        
        return {
            "problem": problem_description,
            "system_info": system_info,
            "solution": result
        }
    
    def code_review(self, code: str, language: str = "python", 
                    focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Review code and provide feedback.
        
        Args:
            code: Code to review
            language: Programming language
            focus_areas: Specific areas to focus on (performance, security, style, etc.)
            
        Returns:
            Code review results
        """
        focus_str = ", ".join(focus_areas) if focus_areas else "general quality"
        
        task = f"""
        Review this {language} code focusing on {focus_str}:
        
        ```{language}
        {code}
        ```
        
        Provide:
        - Code quality assessment
        - Identified issues
        - Improvement suggestions
        - Best practices recommendations
        - Refactored version if needed
        """
        
        result = self.run(task)
        
        return {
            "language": language,
            "focus_areas": focus_areas or ["general"],
            "review": result
        }
