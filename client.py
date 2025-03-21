import asyncio
from typing import Optional, Dict, List, Any
from contextlib import AsyncExitStack
from dataclasses import dataclass
import json
import os
import sys
import time

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

load_dotenv()  # load environment variables from .env

@dataclass
class ResearchResult:
    """Container for research results from various sources"""
    academic_papers: List[Dict]
    web_content: List[Dict]
    scraped_data: List[Dict]
    github_data: List[Dict]  # New field for GitHub data
    summary: str

class MCPClient:
    def __init__(self):
        # Validate required environment variables
        self._validate_env_variables()
        
        # Initialize session and client objects
        self.session: Dict[str, Optional[ClientSession]] = {
            'brave': None,
            'puppeteer': None,
            'notion': None,
            'github': None
        }
        self.exit_stack = AsyncExitStack()
        
        # Initialize Anthropic client
        try:
            self.anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            print("Anthropic client initialized successfully")
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
        
        # Store tools from each server
        self.server_tools: Dict[str, List[Tool]] = {
            'brave': [],
            'puppeteer': [],
            'notion': [],
            'github': []
        }
        
        # Initialize Chroma client
        try:
            persist_directory = os.environ.get("CHROMA_PERSIST_DIRECTORY", "research_db")
            
            # Use the all-MiniLM-L6-v2 embedding function from sentence-transformers
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Create client with updated configuration approach
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory
            )
            
            print(f"ChromaDB client initialized with persist directory: {persist_directory}")
            
            # Create or get collection with embedding function
            self.collection = self.chroma_client.get_or_create_collection(
                name="research_data",
                embedding_function=embedding_function,
                metadata={"description": "Research data from Unified Research Assistant"}
            )
            
            print("ChromaDB collection 'research_data' created or retrieved successfully")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    def _validate_env_variables(self):
        """Validate that all required environment variables are set"""
        required_vars = {
            "ANTHROPIC_API_KEY": "Anthropic API key for Claude",
            "BRAVE_API_KEY": "Brave API key for search",
            "NOTION_API_KEY": "Notion token for integration",
            "GITHUB_API_KEY": "GitHub API key for repository search"
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            if not os.environ.get(var):
                missing_vars.append(f"{var} ({description})")
        
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please add these variables to your .env file or environment")
            sys.exit(1)
        
        # Check optional variables and provide warnings
        if not os.environ.get("NOTION_PAGE_ID"):
            print("Warning: NOTION_PAGE_ID is not set. Will create new pages for each research session.")

    async def connect_to_server(self, server_name: str, server_script_path: str):
        """Connect to an MCP server and list available tools

        Args:
            server_name: Name of the server (brave, puppeteer, notion, github)
            server_script_path: Path to the server script
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=os.environ
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session[server_name] = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session[server_name].initialize()

        # List available tools
        response = await self.session[server_name].list_tools()
        self.server_tools[server_name] = response.tools
        
        print(f"\nConnected to {server_name} server with tools:")
        for tool in self.server_tools[server_name]:
            print(f"  - {tool.name}: {tool.description}")
            if hasattr(tool, 'parameters') and tool.parameters:
                print(f"    Parameters: {tool.parameters}")

    def _create_anthropic_tool_schema(self, server_name: str) -> List[Dict[str, Any]]:
        """Create tool schema for Anthropic tool calling based on MCP server tools"""
        tool_schemas = []
        
        for tool in self.server_tools[server_name]:
            # Convert MCP tool to Anthropic tool schema format
            tool_schema = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",  # Always set type to object as required by Anthropic
                    "properties": {},
                    "required": []
                }
            }
            
            # Add parameters if available
            if hasattr(tool, 'parameters') and tool.parameters:
                try:
                    # If parameters is a JSON string, parse it
                    if isinstance(tool.parameters, str):
                        params = json.loads(tool.parameters)
                    else:
                        params = tool.parameters
                    
                    # Add properties from parameters
                    if "properties" in params:
                        tool_schema["input_schema"]["properties"] = params["properties"]
                    
                    # Add required fields if available
                    if "required" in params:
                        tool_schema["input_schema"]["required"] = params["required"]
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    print(f"Error parsing parameters for {tool.name}: {e}")
            
            tool_schemas.append(tool_schema)
        
        return tool_schemas

    async def search_web(self, query: str) -> List[Dict]:
        """Search web using Brave MCP server via Anthropic tool calling"""
        if not self.session['brave']:
            raise RuntimeError("Brave server not connected")
        
        try:
            # Create tool schema for Anthropic
            brave_tools = self._create_anthropic_tool_schema('brave')
            
            # Create the prompt for Claude with stronger guidance to ensure tool selection
            messages = [{
                "role": "user",
                "content": f"I need to search for information about: '{query}'. You MUST use one of the available Brave search tools to find relevant results. This search is required to continue with research, so tool selection is necessary."
            }]
            
            # Call Claude with tool use
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=messages,
                tools=brave_tools
            )
            
            # Process tool calls from Claude
            search_results = []
            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    
                    print(f"Claude is calling Brave tool: {tool_name} with input: {tool_input}")
                    
                    # Execute the actual MCP tool call
                    result = await self.session['brave'].call_tool(
                        tool_name,
                        tool_input
                    )
                    
                    search_results.append(result.content)
            
            # No fallback - pure MCP approach requires Claude to select a tool
            if not search_results:
                print("No tool calls made by Claude for search. Unable to proceed with research.")
                raise RuntimeError("Claude did not select any search tools. Research cannot continue without search results.")
            
            # Flatten results if needed
            flattened_results = []
            for result in search_results:
                if isinstance(result, list):
                    flattened_results.extend(result)
                else:
                    flattened_results.append(result)
            
            return flattened_results
        except Exception as e:
            print(f"Error during web search: {e}")
            raise

    async def scrape_content(self, urls: List[str]) -> List[Dict]:
        """Scrape content using Puppeteer MCP server via Anthropic tool calling"""
        if not self.session['puppeteer']:
            raise RuntimeError("Puppeteer server not connected")
        
        # Create tool schema for Anthropic
        puppeteer_tools = self._create_anthropic_tool_schema('puppeteer')
        
        scraped_data = []
        for url in urls:
            # Create the prompt for Claude with stronger guidance to ensure tool selection
            messages = [{
                "role": "user",
                "content": f"I need to extract content from this URL: '{url}'. You MUST select and use one of the available Puppeteer tools to scrape this content. Consider what information would be most valuable to extract from this page and choose the most appropriate tool. The URL must be scraped, so tool selection is required."
            }]
            
            # Call Claude with tool use
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=messages,
                tools=puppeteer_tools
            )
            
            # Process tool calls from Claude
            url_data = []
            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    
                    print(f"Claude is calling Puppeteer tool: {tool_name} with input: {tool_input}")
                    
                    # Add URL to tool input if not already present
                    if "url" not in tool_input:
                        tool_input["url"] = url
                    
                    # Execute the actual MCP tool call
                    result = await self.session['puppeteer'].call_tool(
                        tool_name,
                        tool_input
                    )
                    
                    url_data.append(result.content)
            
            # No fallback - if Claude doesn't make a tool selection, we don't scrape this URL
            if not url_data:
                print(f"No tool calls made by Claude for URL: {url}, skipping this URL")
                continue
            
            # Add metadata about the URL
            for data in url_data:
                if isinstance(data, dict):
                    data["source_url"] = url
                else:
                    # If data is not a dict, wrap it
                    data = {"content": data, "source_url": url}
                
                scraped_data.append(data)
        
        return scraped_data

    async def github_search(self, query: str) -> List[Dict]:
        """Search GitHub repositories and pull requests related to the query"""
        if not self.session['github']:
            raise RuntimeError("GitHub server not connected")
        
        try:
            # Create tool schema for Anthropic
            github_tools = self._create_anthropic_tool_schema('github')
            print(f"Created {len(github_tools)} GitHub tool schemas for Anthropic")
            
            # Log available tools
            for i, tool in enumerate(github_tools):
                print(f"GitHub tool {i+1}: {tool['name']} - {tool['description'][:50]}...")
            
            # Create the prompt for Claude with guidance to ensure tool selection
            messages = [{
                "role": "user",
                "content": f"I need to find GitHub repositories related to: '{query}'. You MUST use one of the available GitHub tools to search for relevant repositories. Focus on finding repositories that are directly related to this topic, especially those with active development or significant stars."
            }]
            
            print(f"Sending GitHub search request to Claude for query: '{query}'")
            
            # Call Claude with tool use
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=messages,
                tools=github_tools
            )
            
            print(f"Received response from Claude with {len(response.content)} content items")
            
            # Process tool calls from Claude
            github_results = []
            tool_calls = 0
            
            for i, content in enumerate(response.content):
                print(f"Content item {i+1} type: {content.type}")
                
                if content.type == "tool_use":
                    tool_calls += 1
                    tool_name = content.name
                    tool_input = content.input
                    
                    print(f"Claude is calling GitHub tool: {tool_name} with input: {tool_input}")
                    
                    # Execute the actual MCP tool call
                    print(f"Executing MCP tool call to GitHub server: {tool_name}")
                    result = await self.session['github'].call_tool(
                        tool_name,
                        tool_input
                    )
                    
                    print(f"Received result from GitHub server for {tool_name}")
                    
                    # Process the result - handle TextContent objects
                    processed_result = None
                    if hasattr(result, 'content'):
                        result_content = result.content
                        result_type = type(result_content)
                        print(f"Result type: {result_type}")
                        
                        # Handle list of TextContent objects
                        if isinstance(result_content, list):
                            print(f"Result is a list with {len(result_content)} items")
                            
                            # Process each item in the list
                            processed_items = []
                            for item in result_content:
                                if hasattr(item, 'text') and hasattr(item, 'type') and item.type == 'text':
                                    # Parse JSON from TextContent
                                    try:
                                        import json
                                        json_data = json.loads(item.text)
                                        print(f"Successfully parsed JSON from TextContent")
                                        
                                        # For search_repositories, extract the items array
                                        if tool_name == 'search_repositories' and 'items' in json_data:
                                            processed_items.extend(json_data['items'])
                                            print(f"Extracted {len(json_data['items'])} repositories from search results")
                                        else:
                                            processed_items.append(json_data)
                                    except json.JSONDecodeError as e:
                                        print(f"Failed to parse JSON from TextContent: {e}")
                                        processed_items.append({"error": "Failed to parse JSON", "text": str(item.text)[:100]})
                                else:
                                    processed_items.append(item)
                            
                            processed_result = processed_items
                        # Handle single TextContent object
                        elif hasattr(result_content, 'text') and hasattr(result_content, 'type') and result_content.type == 'text':
                            try:
                                import json
                                json_data = json.loads(result_content.text)
                                print(f"Successfully parsed JSON from TextContent")
                                
                                # For search_repositories, extract the items array
                                if tool_name == 'search_repositories' and 'items' in json_data:
                                    processed_result = json_data['items']
                                    print(f"Extracted {len(json_data['items'])} repositories from search results")
                                else:
                                    processed_result = json_data
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse JSON from TextContent: {e}")
                                processed_result = {"error": "Failed to parse JSON", "text": str(result_content.text)[:100]}
                        else:
                            processed_result = result_content
                    else:
                        print(f"Result has no content attribute: {result}")
                        processed_result = result
                    
                    # Store both the tool name and processed result
                    github_results.append({
                        "tool": tool_name,
                        "result": processed_result
                    })
                    print(f"Added result from {tool_name} to github_results (now {len(github_results)} items)")
            
            print(f"Processed {tool_calls} tool calls from Claude")
            
            # If no results, provide a clear message
            if not github_results:
                print("No GitHub tool calls made by Claude. Unable to find GitHub repositories.")
                return []
            
            print(f"Returning {len(github_results)} GitHub results")
            return github_results
        except Exception as e:
            print(f"Error during GitHub search: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def save_to_notion(self, research_data: ResearchResult):
        """Save research results to Notion using the Notion MCP server"""
        if not self.session['notion']:
            print("Notion server not connected. Skipping save to Notion.")
            return None
        
        try:
            # Format the research data as markdown for Notion
            content_markdown = f"# Research Summary\n\n{research_data.summary}\n\n"
            
            # Add web content if available
            if research_data.web_content:
                content_markdown += "## Web Search Results\n\n"
                for i, result in enumerate(research_data.web_content[:5], 1):
                    if hasattr(result, 'type') and result.type == 'text':
                        # Handle TextContent objects
                        content_markdown += f"### {i}. Web Result\n\n"
                        content_markdown += f"{result.text[:500]}...\n\n"
                    elif isinstance(result, dict):
                        title = result.get("title", f"Result {i}")
                        url = result.get("url", "#")
                        snippet = result.get("snippet", "")
                        
                        content_markdown += f"### {i}. [{title}]({url})\n\n"
                        content_markdown += f"{snippet}\n\n"
                    else:
                        content_markdown += f"### {i}. Web Result\n\n"
                        content_markdown += f"{str(result)[:500]}...\n\n"
            
            # Add GitHub data if available
            if research_data.github_data:
                content_markdown += "## GitHub Repositories\n\n"
                
                # Check if we have actual repositories
                has_repos = False
                
                for i, item in enumerate(research_data.github_data[:3], 1):
                    if not isinstance(item, dict):
                        continue
                        
                    tool_name = item.get('tool', 'Unknown tool')
                    result = item.get('result', [])
                    
                    content_markdown += f"### {i}. {tool_name} Results\n\n"
                    
                    # Handle different result types
                    if isinstance(result, list) and result:
                        has_repos = True
                        for j, repo in enumerate(result[:3], 1):
                            if isinstance(repo, dict):
                                name = repo.get('full_name', repo.get('name', f"Repository {j}"))
                                url = repo.get('html_url', repo.get('url', '#'))
                                desc = repo.get('description', 'No description')
                                stars = repo.get('stargazers_count', 'Unknown')
                                
                                content_markdown += f"#### {j}. [{name}]({url})\n\n"
                                content_markdown += f"**Stars:** {stars}\n\n"
                                content_markdown += f"**Description:** {desc}\n\n"
                    elif isinstance(result, dict) and result:
                        has_repos = True
                        name = result.get('full_name', result.get('name', 'Repository'))
                        url = result.get('html_url', result.get('url', '#'))
                        desc = result.get('description', 'No description')
                        stars = result.get('stargazers_count', 'Unknown')
                        
                        content_markdown += f"#### [{name}]({url})\n\n"
                        content_markdown += f"**Stars:** {stars}\n\n"
                        content_markdown += f"**Description:** {desc}\n\n"
                    else:
                        content_markdown += "No repositories found matching the search criteria.\n\n"
                
                if not has_repos:
                    content_markdown += "No repositories found matching the search criteria.\n\n"
            
            # Prepare page content
            page_content = {
                "title": "Research Results",
                "content": content_markdown
            }
            
            # Add page_id if available
            notion_page_id = os.environ.get("NOTION_PAGE_ID")
            if notion_page_id:
                page_content["parentPageId"] = notion_page_id
                print(f"Saving research results to existing Notion page: {notion_page_id}")
            else:
                print("Creating new Notion page for research results")
            
            print(f"Sending to Notion with parentPageId: {json.dumps(page_content.get('parentPageId', 'none'), default=str)}")
            print(f"Content length: {len(page_content['content'])} characters")
            
            # Create a new page with research results
            result = await self.session['notion'].call_tool(
                "create_page",
                page_content
            )
            
            print(f"Successfully saved research results to Notion: {result.content}")
            return result.content
        except Exception as e:
            print(f"Error saving to Notion: {e}")
            print(f"Full error: {json.dumps(str(e), default=str)}")
            print("Research results will not be available in Notion")
            return None

    def store_in_chroma(self, research_data: ResearchResult) -> bool:
        """Store research results in Chroma vector database"""
        try:
            # Generate a unique timestamp for this research session
            timestamp = str(int(time.time()))
            
            # Simple string conversion that avoids JSON serialization issues
            def simple_str(obj):
                if hasattr(obj, 'text') and hasattr(obj, 'type') and obj.type == 'text':
                    return str(obj.text)
                elif hasattr(obj, 'text') and callable(getattr(obj, 'text', None)):
                    return str(obj.text())
                else:
                    return str(obj)
            
            # First, delete any existing entries to avoid ID conflicts
            try:
                self.collection.delete(where={"timestamp": {"$exists": True}})
                print("Cleared existing entries from ChromaDB")
            except Exception as e:
                print(f"Note: Could not clear existing entries: {e}")
            
            # Store all data in a single batch with unique IDs
            docs = []
            metadatas = []
            ids = []
            
            # Process GitHub data
            print(f"Processing GitHub data for ChromaDB storage: {len(research_data.github_data)} items")
            for i, data in enumerate(research_data.github_data):
                try:
                    # Debug logging
                    print(f"GitHub data item {i+1} type: {type(data)}")
                    print(f"GitHub data item {i+1} keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
                    
                    # Create a simple document string
                    doc = f"GitHub data {i+1}"
                    if isinstance(data, dict):
                        tool = data.get('tool', 'unknown')
                        print(f"  Tool: {tool}")
                        doc = f"Tool: {tool}\n"
                        
                        # Handle result based on type
                        result = data.get('result', '')
                        print(f"  Result type: {type(result)}")
                        
                        if isinstance(result, list):
                            print(f"  Result is a list with {len(result)} items")
                            doc += f"Results: {len(result)} items found\n"
                            # Log first item if available
                            if result and isinstance(result[0], dict):
                                print(f"  First result item keys: {result[0].keys()}")
                        elif isinstance(result, dict):
                            print(f"  Result is a dict with keys: {result.keys()}")
                            doc += f"Result: {result.get('name', 'Unknown')}\n"
                        else:
                            print(f"  Result is a {type(result)}")
                            doc += f"Result: {simple_str(result)}\n"
                    
                    docs.append(doc)
                    metadatas.append({
                        "type": "github",
                        "source": "github",
                        "timestamp": timestamp
                    })
                    ids.append(f"github_{timestamp}_{i}")
                    print(f"  Successfully processed GitHub item {i+1}")
                except Exception as e:
                    print(f"Skipping GitHub item {i+1} due to error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Process web content
            print(f"Processing web content for ChromaDB storage: {len(research_data.web_content)} items")
            for i, content in enumerate(research_data.web_content):
                try:
                    # Debug logging
                    print(f"Web content item {i+1} type: {type(content)}")
                    
                    # Create a simple document string
                    doc = f"Web content {i+1}"
                    if isinstance(content, dict):
                        title = content.get('title', 'No title')
                        url = content.get('url', 'No URL')
                        snippet = content.get('snippet', 'No snippet')
                        print(f"  Title: {title[:30]}...")
                        doc = f"Title: {title}\nURL: {url}\nSnippet: {snippet}"
                    
                    docs.append(doc)
                    metadatas.append({
                        "type": "web",
                        "source": "brave",
                        "timestamp": timestamp
                    })
                    ids.append(f"web_{timestamp}_{i}")
                    print(f"  Successfully processed web content item {i+1}")
                except Exception as e:
                    print(f"Skipping web content item {i+1} due to error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Add all documents in a single batch if we have any
            if docs:
                print(f"Adding {len(docs)} documents to ChromaDB with IDs: {ids}")
                self.collection.add(
                    documents=docs,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Stored {len(docs)} total items in ChromaDB")
            else:
                print("No documents to store in ChromaDB")
            
            print("ChromaDB data stored successfully")
            return True
        except Exception as e:
            print(f"Error storing data in ChromaDB: {e}")
            import traceback
            traceback.print_exc()
            print("Research results will not be available for future queries")
            return False

    async def generate_summary(self, research_data: ResearchResult) -> str:
        """Generate a comprehensive summary of all research data using Claude"""
        try:
            # Log the data we're working with
            print(f"Generating summary with: {len(research_data.web_content)} web results, " +
                  f"{len(research_data.scraped_data)} scraped items, " +
                  f"{len(research_data.github_data)} GitHub items")
            
            # Check if we have any actual data to summarize
            has_data = False
            
            # Prepare the data for Claude
            web_content_summary = ""
            if research_data.web_content:
                has_data = True
                web_content_summary = "\n\n".join([
                    f"Title: {item.get('title', 'No title') if isinstance(item, dict) else 'No title'}\n" +
                    f"URL: {item.get('url', 'No URL') if isinstance(item, dict) else 'No URL'}\n" +
                    f"Snippet: {item.get('snippet', 'No snippet') if isinstance(item, dict) else str(item)[:300]}"
                    for item in research_data.web_content[:5]  # Limit to first 5 items
                ])
            
            scraped_content_summary = ""
            if research_data.scraped_data:
                has_data = True
                scraped_content_summary = "\n\n".join([
                    f"Title: {item.get('title', 'No title') if isinstance(item, dict) else 'No title'}\n" +
                    f"URL: {item.get('source_url', 'No URL') if isinstance(item, dict) else 'No URL'}\n" +
                    f"Content: {item.get('content', str(item))[:500]}..."  # Truncate long content
                    for item in research_data.scraped_data[:3]  # Limit to first 3 items
                ])
            
            # Process GitHub data in a more flexible way
            github_content_summary = ""
            has_github_data = False
            print(f"Processing {len(research_data.github_data)} GitHub items for summary")
            
            for i, item in enumerate(research_data.github_data[:3]):  # Limit to first 3 items
                print(f"GitHub item {i+1} type: {type(item)}")
                if isinstance(item, dict):
                    print(f"GitHub item {i+1} keys: {item.keys()}")
                
                tool_name = item.get('tool', 'Unknown tool') if isinstance(item, dict) else 'Unknown tool'
                result = item.get('result', {}) if isinstance(item, dict) else {}
                
                print(f"Tool: {tool_name}, Result type: {type(result)}")
                github_content_summary += f"\nTool: {tool_name}\n"
                
                # Handle different result types
                if isinstance(result, list) and result:
                    has_github_data = True
                    has_data = True
                    print(f"Result is a list with {len(result)} items")
                    github_content_summary += f"Found {len(result)} items\n"
                    for j, res_item in enumerate(result[:3]):  # Show first 3 items
                        if isinstance(res_item, dict):
                            print(f"  Item {j+1} keys: {res_item.keys()}")
                            name = res_item.get('full_name', res_item.get('name', f"Item {j+1}"))
                            desc = res_item.get('description', 'No description')
                            github_content_summary += f"- {name}: {desc}\n"
                elif isinstance(result, dict) and result:
                    has_github_data = True
                    has_data = True
                    # Single item result
                    print(f"Result is a dict with keys: {result.keys()}")
                    name = result.get('full_name', result.get('name', 'Item'))
                    desc = result.get('description', 'No description')
                    github_content_summary += f"- {name}: {desc}\n"
                else:
                    # Handle text or other content
                    print(f"Result is a {type(result)}")
                    github_content_summary += "No repositories found matching the search criteria.\n"
            
            if not has_github_data:
                github_content_summary += "No GitHub repositories found matching the search criteria.\n"
            
            print("GitHub content summary created")
            
            # If we have no data at all, return a message
            if not has_data:
                return "Unfortunately, without any web search results, scraped content, or GitHub repositories provided, I do not have enough information to create a comprehensive research summary. A meaningful summary requires source material to analyze and synthesize key findings, concepts, developments, areas of consensus/disagreement, and potential areas for further research. Please provide the relevant web pages, documents, code repositories, etc. that you would like me to review and summarize. I'd be happy to generate a detailed research summary once I have the necessary source content."
            
            # Create the prompt for Claude
            prompt = f"""
            Please create a comprehensive research summary based on the following information:
            
            WEB SEARCH RESULTS:
            {web_content_summary}
            
            SCRAPED CONTENT:
            {scraped_content_summary}
            
            GITHUB REPOSITORIES AND PULL REQUESTS:
            {github_content_summary}
            
            Your summary should:
            1. Synthesize the key findings and information
            2. Highlight important concepts, trends, and developments
            3. Note any significant GitHub projects and active development
            4. Identify areas of consensus and disagreement
            5. Suggest areas for further research
            
            Format the summary with clear sections and bullet points where appropriate.
            """
            
            print("Sending summary request to Claude")
            
            # Call Claude for the summary
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            print("Received summary from Claude")
            return response.content[0].text
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating summary: {str(e)}"

    async def research_topic(self, query: str) -> ResearchResult:
        """Main method to research a topic using all available tools"""
        # Check if all required servers are connected
        if not self.session['brave']:
            raise RuntimeError("Brave server not connected. Cannot perform web search.")
        
        if not self.session['puppeteer']:
            print("Warning: Puppeteer server not connected. Skipping content scraping.")
        
        if not self.session['github']:
            print("Warning: GitHub server not connected. Skipping GitHub research.")
        
        if not self.session['notion']:
            print("Warning: Notion server not connected. Results will not be saved to Notion.")
        
        try:
            print(f"\nResearching topic: '{query}'")
            print("Step 1/5: Searching web content...")
            
            # Search web content using Anthropic tool calling
            web_content = await self.search_web(query)
            print(f"Found {len(web_content)} web results")
            
            # Extract URLs for deeper scraping if Puppeteer is connected
            scraped_data = []
            if self.session['puppeteer']:
                print("\nStep 2/5: Extracting URLs for detailed content scraping...")
                urls = [item['url'] for item in web_content if 'url' in item][:5]  # Limit to first 5 URLs
                
                if urls:
                    print(f"Scraping content from {len(urls)} URLs...")
                    scraped_data = await self.scrape_content(urls)
                    print(f"Successfully scraped {len(scraped_data)} content items")
                else:
                    print("No valid URLs found for scraping")
            else:
                print("Skipping content scraping (Puppeteer server not connected)")
            
            # Search GitHub if connected
            github_data = []
            if self.session['github']:
                print("\nStep 3/5: Searching GitHub repositories and PRs...")
                github_data = await self.github_search(query)
                print(f"Found {len(github_data)} GitHub repositories with PRs")
            else:
                print("Skipping GitHub research (GitHub server not connected)")
            
            # Create research result object
            research_data = ResearchResult(
                academic_papers=[],  # Would come from academic MCP server
                web_content=web_content,
                scraped_data=scraped_data,
                github_data=github_data,
                summary=""
            )
            
            # Generate summary
            print("\nStep 4/5: Generating comprehensive summary...")
            research_data.summary = await self.generate_summary(research_data)
            print("Summary generated successfully")
            
            # Store in Chroma
            print("\nStep 5/5: Storing research data...")
            store_success = self.store_in_chroma(research_data)
            
            # Save to Notion if connected
            if self.session['notion']:
                print("Saving results to Notion...")
                await self.save_to_notion(research_data)
            
            print("\nResearch complete!")
            return research_data
            
        except Exception as e:
            print(f"Error during research process: {e}")
            # Return partial results if available
            if 'research_data' in locals():
                print("Returning partial research results")
                return research_data
            else:
                # Create empty result
                print("Unable to obtain research results")
                return ResearchResult(
                    academic_papers=[],
                    web_content=[],
                    scraped_data=[],
                    github_data=[],
                    summary=f"Research failed with error: {str(e)}"
                )

    async def cleanup(self):
        """Clean up resources"""
        try:
            # PersistentClient automatically persists data, no need to call persist()
            
            # Close async context managers
            await self.exit_stack.aclose()
            print("All resources cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            print("Some resources may not have been properly released")