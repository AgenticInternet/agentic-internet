"""
Example MCP Server for testing AgenticInternet MCP integration.

This server provides simple tools that can be accessed via MCP protocol.
"""

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "FastMCP is required for this example. Install it with: pip install fastmcp"
    )


# Create the MCP server
mcp = FastMCP(name="Example MCP Server")


@mcp.tool
def calculate_sum(a: int, b: int) -> int:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b


@mcp.tool
def calculate_product(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of a and b
    """
    return a * b


@mcp.tool
def get_weather(city: str) -> dict[str, str]:
    """
    Get mock weather information for a city.
    
    Args:
        city: Name of the city
        
    Returns:
        Dictionary with weather information
    """
    # Mock weather data
    weather_data = {
        "city": city,
        "temperature": "72°F",
        "condition": "Sunny",
        "humidity": "45%",
        "wind_speed": "10 mph"
    }
    return weather_data


@mcp.tool
def analyze_sentiment(text: str) -> dict[str, str]:
    """
    Perform basic sentiment analysis on text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Simple mock sentiment analysis
    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "horrible", "poor"]

    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "text": text,
        "sentiment": sentiment,
        "positive_count": str(positive_count),
        "negative_count": str(negative_count)
    }


@mcp.tool
def search_items(query: str, limit: int = 5) -> list[dict[str, str]]:
    """
    Search for items based on a query (mock data).
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of search results
    """
    # Mock search results
    results = [
        {
            "title": f"Result {i+1} for '{query}'",
            "description": f"This is a mock description for search result {i+1}",
            "url": f"https://example.com/results/{i+1}"
        }
        for i in range(min(limit, 10))
    ]
    return results


@mcp.resource("config://settings")
def get_server_config() -> dict[str, str]:
    """
    Provide server configuration information.
    
    Returns:
        Server configuration dictionary
    """
    return {
        "version": "1.0.0",
        "name": "Example MCP Server",
        "author": "AgenticInternet",
        "description": "Example MCP server for testing integration"
    }


@mcp.resource("data://{dataset_name}")
def get_dataset(dataset_name: str) -> dict[str, any]:
    """
    Provide mock dataset information.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dataset information
    """
    datasets = {
        "users": {
            "name": "users",
            "records": 1000,
            "columns": ["id", "name", "email", "created_at"]
        },
        "products": {
            "name": "products",
            "records": 500,
            "columns": ["id", "name", "price", "category"]
        },
        "orders": {
            "name": "orders",
            "records": 2500,
            "columns": ["id", "user_id", "product_id", "quantity", "order_date"]
        }
    }

    return datasets.get(dataset_name, {"error": f"Dataset '{dataset_name}' not found"})


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
