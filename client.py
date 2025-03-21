import asyncio
from typing import Optional, Dict, List, Any
from contextlib import AsyncExitStack
from dataclasses import dataclass
import json
import os
import sys
import time
import websockets

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
        self._validate_env_variables()
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.event_queue = []  # Queue for events before WebSocket connects
        self.session: Dict[str, Optional[ClientSession]] = {
            'brave': None, 'puppeteer': None, 'notion': None, 'github': None
        }
        self.exit_stack = AsyncExitStack()
        self.server_tools: Dict[str, List[Tool]] = {
            'brave': [], 'puppeteer': [], 'notion': [], 'github': []
        }
        try:
            self.anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            print("Anthropic client initialized successfully")
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
        try:
            persist_directory = os.environ.get("CHROMA_PERSIST_DIRECTORY", "research_db")
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.chroma_client.get_or_create_collection(
                name="research_data", embedding_function=embedding_function,
                metadata={"description": "Research data from Unified Research Assistant"}
            )
            print(f"ChromaDB client initialized with persist directory: {persist_directory}")
            print("ChromaDB collection 'research_data' created or retrieved successfully")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    def _validate_env_variables(self):
        required_vars = {
            "ANTHROPIC_API_KEY": "Anthropic API key for Claude",
            "BRAVE_API_KEY": "Brave API key for search",
            "NOTION_API_KEY": "Notion token for integration",
            "GITHUB_API_KEY": "GitHub API key for repository search"
        }
        missing_vars = [f"{var} ({desc})" for var, desc in required_vars.items() if not os.environ.get(var)]
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            sys.exit(1)
        if not os.environ.get("NOTION_PAGE_ID"):
            print("Warning: NOTION_PAGE_ID is not set. Will create new pages for each research session.")

    async def send_event(self, event_type: str, message: str, source: str = None, target: str = None):
        event = {"type": event_type, "message": message}
        if source:
            event["source"] = source
        if target:
            event["target"] = target
        if self.websocket is not None:
            try:
                print(f"Sending WebSocket event: {json.dumps(event)}")
                await self.websocket.send(json.dumps(event))
            except websockets.ConnectionClosed:
                print("WebSocket connection closed while sending event")
                self.websocket = None
        else:
            print(f"Queueing event (WebSocket not connected): {json.dumps(event)}")
            self.event_queue.append(event)
    
    async def flush_event_queue(self):
        if self.websocket is not None:
            try:
                for event in self.event_queue:
                    print(f"Sending queued WebSocket event: {json.dumps(event)}")
                    await self.websocket.send(json.dumps(event))
                self.event_queue.clear()
            except websockets.ConnectionClosed:
                print("WebSocket connection closed while flushing queue")
                self.websocket = None

    async def connect_to_server(self, server_name: str, server_script_path: str):
        await self.send_event("status", f"Connecting to {server_name} server")
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=os.environ)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session[server_name] = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session[server_name].initialize()
        response = await self.session[server_name].list_tools()
        self.server_tools[server_name] = response.tools
        await self.send_event("connection", f"Connected to {server_name}", "Research Assistant", server_name.capitalize())
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
        if not self.session['brave']:
            raise RuntimeError("Brave server not connected")
        await self.send_event("data", "Searching web via Brave", "Brave", "Research Assistant")
        try:
            brave_tools = self._create_anthropic_tool_schema('brave')
            messages = [{
                "role": "user",
                "content": f"Search for '{query}'. Use a Brave tool to return results with URLs. Tool use is mandatory."
            }]
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=messages,
                tools=brave_tools
            )
            search_results = []
            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    print(f"Claude is calling Brave tool: {tool_name} with input: {tool_input}")
                    result = await self.session['brave'].call_tool(tool_name, tool_input)
                    if hasattr(result, 'content'):
                        if isinstance(result.content, list):
                            search_results.extend(result.content)
                            
                        elif isinstance(result.content, str):
                        # Parse string result into list of dicts
                            lines = result.content.split('\n\n')
                            for line in lines:
                                if 'URL:' in line:
                                    title = line.split('Title: ')[1].split('\n')[0] if 'Title: ' in line else "Untitled"
                                    url = line.split('URL: ')[1].strip() if 'URL: ' in line else None
                                    desc = line.split('Description: ')[1].split('\n')[0] if 'Description: ' in line else ""
                                    if url:
                                        search_results.append({'title': title, 'url': url, 'snippet': desc})
                        else:
                            search_results.append(result.content)
            
            print(f"Raw Brave results: {search_results}")
            formatted_results = [item for item in search_results if isinstance(item, dict) and 'url' in item]
            print(f"Formatted web results: {formatted_results}")
            return formatted_results
        
        except Exception as e:
            print(f"Error during web search: {e}")
        raise

    async def scrape_content(self, urls: List[str]) -> List[Dict]:
        if not self.session['puppeteer']:
            raise RuntimeError("Puppeteer server not connected")
        await self.send_event("data", f"Scraping {len(urls)} URLs", "Puppeteer", "Research Assistant")
        puppeteer_tools = self._create_anthropic_tool_schema('puppeteer')
        scraped_data = []
        for url in urls:
            print(f"Attempting to scrape: {url}")
            messages = [{
                "role": "user",
                "content": f"Scrape meaningful content from '{url}'. You MUST use a Puppeteer tool (e.g., puppeteer_navigate followed by puppeteer_evaluate) to extract text or metadata. Tool use is mandatory for this URL."
            }]
            try:
                response = self.anthropic.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    messages=messages,
                    tools=puppeteer_tools
                )
                url_data = []
                for content in response.content:
                    if content.type == "tool_use":
                        tool_name = content.name
                        tool_input = content.input
                        if "url" not in tool_input:
                            tool_input["url"] = url
                        print(f"Claude is calling Puppeteer tool: {tool_name} with input: {tool_input}")
                        result = await self.session['puppeteer'].call_tool(tool_name, tool_input)
                        url_data.append({"source_url": url, "content": result.content})
                        await self.send_event("data", f"Scraped {url} with {tool_name}", "Puppeteer", "Research Assistant")
                if not url_data:
                    print(f"No Puppeteer tool selected by Claude for {url}")
                    await self.send_event("status", f"Failed to scrape {url}: No tool selected", "Puppeteer")
                scraped_data.extend(url_data)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                await self.send_event("status", f"Error scraping {url}: {str(e)}", "Puppeteer")
        await self.send_event("status", f"Scraped {len(scraped_data)} items")
        return scraped_data

    async def github_search(self, query: str) -> List[Dict]:
        """Search GitHub repositories and pull requests related to the query"""
        if not self.session['github']:
            raise RuntimeError("GitHub server not connected")
        await self.send_event("data", "Searching GitHub", "GitHub", "Research Assistant")
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
            
            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_input = content.input
                    print(f"Claude is calling GitHub tool: {tool_name} with input: {tool_input}")
                    result = await self.session['github'].call_tool(tool_name, tool_input)
                    processed_result = None
                    if hasattr(result, 'content'):
                        result_content = result.content
                        if isinstance(result_content, list):
                            processed_items = []
                            for item in result_content:
                                if hasattr(item, 'text') and item.type == 'text':
                                    try:
                                        json_data = json.loads(item.text)
                                        if tool_name == 'search_repositories' and 'items' in json_data:
                                            processed_items.extend(json_data['items'])
                                        else:
                                            processed_items.append(json_data)
                                    except json.JSONDecodeError:
                                        processed_items.append({"error": "Failed to parse JSON", "text": item.text[:100]})
                                else:
                                    processed_items.append(item)
                            processed_result = processed_items
                        elif hasattr(result_content, 'text') and result_content.type == 'text':
                            try:
                                json_data = json.loads(result_content.text)
                                processed_result = json_data['items'] if tool_name == 'search_repositories' and 'items' in json_data else json_data
                            except json.JSONDecodeError:
                                processed_result = {"error": "Failed to parse JSON", "text": result_content.text[:100]}
                        else:
                            processed_result = result_content
                    else:
                        processed_result = result
                    github_results.append({"tool": tool_name, "result": processed_result})
            await self.send_event("status", f"Found {len(github_results)} GitHub results")
            return github_results if github_results else []
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
        await self.send_event("data", "Saving to Notion", "Research Assistant", "Notion")
        
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

    async def store_in_chroma(self, research_data: ResearchResult) -> bool:
        await self.send_event("status", "Storing in ChromaDB")
        try:
            timestamp = str(int(time.time()))
            def simple_str(obj):
                if hasattr(obj, 'text') and hasattr(obj, 'type') and obj.type == 'text':
                    return str(obj.text)
                elif hasattr(obj, 'text') and callable(getattr(obj, 'text', None)):
                    return str(obj.text())
                else:
                    return str(obj)
            try:
                self.collection.delete(where={"timestamp": {"$eq": timestamp}})
                print("Cleared existing entries from ChromaDB")
            except Exception as e:
                print(f"Note: Could not clear existing entries: {e}")
            docs, metadatas, ids = [], [], []
            for i, data in enumerate(research_data.github_data):
                try:
                    doc = f"Tool: {data.get('tool', 'unknown')}\n"
                    result = data.get('result', '')
                    if isinstance(result, list) and result:
                        doc += f"Results: {len(result)} items found\n"
                    elif isinstance(result, dict):
                        doc += f"Result: {result.get('name', 'Unknown')}\n"
                    else:
                        doc += f"Result: {simple_str(result)}\n"
                    docs.append(doc)
                    metadatas.append({"type": "github", "source": "github", "timestamp": timestamp})
                    ids.append(f"github_{timestamp}_{i}")
                except Exception as e:
                    print(f"Skipping GitHub item {i+1} due to error: {e}")
            for i, content in enumerate(research_data.web_content):
                try:
                    doc = f"Title: {content.get('title', 'No title')}\nURL: {content.get('url', 'No URL')}\nSnippet: {content.get('snippet', 'No snippet')}" if isinstance(content, dict) else f"Web content {i+1}"
                    docs.append(doc)
                    metadatas.append({"type": "web", "source": "brave", "timestamp": timestamp})
                    ids.append(f"web_{timestamp}_{i}")
                except Exception as e:
                    print(f"Skipping web content item {i+1} due to error: {e}")
            if docs:
                self.collection.add(documents=docs, metadatas=metadatas, ids=ids)
                print(f"Stored {len(docs)} total items in ChromaDB")
            else:
                print("No documents to store in ChromaDB")
            await self.send_event("status", "Stored in ChromaDB")
            return True
        except Exception as e:
            print(f"Error storing data in ChromaDB: {e}")
            return False

    async def generate_summary(self, research_data: ResearchResult) -> str:
        await self.send_event("status", "Generating summary")
        try:
            has_data = False
            web_content_summary = ""
            if research_data.web_content:
                has_data = True
                web_content_summary = "\n\n".join([
                    f"Title: {item.get('title', 'No title') if isinstance(item, dict) else 'No title'}\nURL: {item.get('url', 'No URL') if isinstance(item, dict) else 'No URL'}\nSnippet: {item.get('snippet', 'No snippet') if isinstance(item, dict) else str(item)[:300]}"
                    for item in research_data.web_content[:5]
                ])
            scraped_content_summary = ""
            if research_data.scraped_data:
                has_data = True
                scraped_content_summary = "\n\n".join([
                    f"Title: {item.get('title', 'No title') if isinstance(item, dict) else 'No title'}\nURL: {item.get('source_url', 'No URL') if isinstance(item, dict) else 'No URL'}\nContent: {item.get('content', str(item))[:500]}..."
                    for item in research_data.scraped_data[:3]
                ])
            github_content_summary = ""
            has_github_data = False
            for i, item in enumerate(research_data.github_data[:3]):
                tool_name = item.get('tool', 'Unknown tool') if isinstance(item, dict) else 'Unknown tool'
                result = item.get('result', {}) if isinstance(item, dict) else {}
                github_content_summary += f"\nTool: {tool_name}\n"
                if isinstance(result, list) and result:
                    has_github_data = True
                    has_data = True
                    github_content_summary += f"Found {len(result)} items\n"
                    for j, res_item in enumerate(result[:3]):
                        if isinstance(res_item, dict):
                            name = res_item.get('full_name', res_item.get('name', f"Item {j+1}"))
                            desc = res_item.get('description', 'No description')
                            github_content_summary += f"- {name}: {desc}\n"
                elif isinstance(result, dict) and result:
                    has_github_data = True
                    has_data = True
                    name = result.get('full_name', result.get('name', 'Item'))
                    desc = result.get('description', 'No description')
                    github_content_summary += f"- {name}: {desc}\n"
                else:
                    github_content_summary += "No repositories found matching the search criteria.\n"
            if not has_github_data:
                github_content_summary += "No GitHub repositories found matching the search criteria.\n"
            if not has_data:
                return "Unfortunately, without any data provided, I cannot create a comprehensive research summary."
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
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229", max_tokens=4000, messages=[{"role": "user", "content": prompt}]
            )
            await self.send_event("status", "Summary generated")
            return response.content[0].text
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

    async def research_topic(self, query: str) -> ResearchResult:
        if not self.session['brave']:
            raise RuntimeError("Brave server not connected. Cannot perform web search.")
        if not self.session['puppeteer']:
            print("Warning: Puppeteer server not connected. Skipping content scraping.")
        if not self.session['github']:
            print("Warning: GitHub server not connected. Skipping GitHub research.")
        if not self.session['notion']:
            print("Warning: Notion server not connected. Results will not be saved to Notion.")
        
        await self.send_event("status", f"Starting research on '{query}'")
        try:
            print(f"\nResearching topic: '{query}'")
            print("Step 1/5: Searching web content...")
            web_content = await self.search_web(query)
            print(f"Found {len(web_content)} web results")

            scraped_data = []
            if self.session['puppeteer']:
                print("\nStep 2/5: Extracting URLs for detailed content scraping...")
                urls = [item['url'] for item in web_content if isinstance(item, dict) and 'url' in item][:5]
                print(f"Extracted {len(urls)} URLs: {urls}")
                if urls:
                    print(f"Scraping content from {len(urls)} URLs...")
                    scraped_data = await self.scrape_content(urls)
                else:
                    print("No valid URLs found for scraping")
                    await self.send_event("status", "No URLs available for scraping")
            
            github_data = []
            if self.session['github']:
                print("\nStep 3/5: Searching GitHub repositories and PRs...")
                github_data = await self.github_search(query)
            
            research_data = ResearchResult(
                academic_papers=[], web_content=web_content, scraped_data=scraped_data,
                github_data=github_data, summary=""
            )
            
            print("\nStep 4/5: Generating comprehensive summary...")
            research_data.summary = await self.generate_summary(research_data)
            
            print("\nStep 5/5: Storing research data...")
            await self.store_in_chroma(research_data)
            if self.session['notion']:
                print("Saving results to Notion...")
                await self.save_to_notion(research_data)
            
            print("\nResearch complete!")
            await self.send_event("status", "Research completed")
            return research_data
        except Exception as e:
            print(f"Error during research process: {e}")
            if 'research_data' in locals():
                print("Returning partial research results")
                return research_data
            return ResearchResult(
                academic_papers=[], web_content=[], scraped_data=[], github_data=[],
                summary=f"Research failed with error: {str(e)}"
            )
    
    async def run_research(self, websocket, query):
        self.websocket = websocket
        await self.send_event("status", "WebSocket client connected")
        await self.flush_event_queue()  # Send queued events
        await self.research_topic(query)

    

    async def cleanup(self):
        try:
            if self.websocket is not None:
                try:
                    await self.websocket.close()
                    print("WebSocket closed during cleanup")
                except websockets.ConnectionClosed:
                    print("WebSocket already closed during cleanup")
                self.websocket = None
            # Gracefully close sessions
            for server_name, session in self.session.items():
                if session:
                    try:
                        
                        print(f"Closed {server_name} session")
                    except Exception as e:
                        print(f"Error closing {server_name} session: {e}")
            async with asyncio.timeout(5):
                await self.exit_stack.aclose()
            print("All resources cleaned up successfully")
        except asyncio.TimeoutError:
            print("Cleanup timed out, some resources may remain open")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            print("Some resources may not have been properly released")