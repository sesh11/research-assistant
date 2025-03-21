# Unified Research Assistant

A comprehensive research tool that integrates multiple MCP servers with Anthropic's Claude for intelligent information gathering and organization.

## Features

- **Pure MCP Integration**: Connects to Brave, Puppeteer, and Notion MCP servers with full AI-driven tool selection
- **Anthropic Tool Calling**: Uses Claude to intelligently call Brave search and Puppeteer scraping tools
- **Content Scraping**: Extracts detailed information from web pages using intelligent tool selection
- **Vector Storage**: Stores research data in a Chroma vector database with semantic search capabilities
- **Summarization**: Generates comprehensive summaries of research findings
- **Notion Integration**: Organizes and saves research results in Notion

## Prerequisites

- Python 3.10+
- Node.js 16+ (for JavaScript-based MCP servers)
- Anthropic API key (Claude 3 Sonnet model access required)
- Brave Search API key
- Notion API token and workspace access

## Detailed Setup

### 1. Environment Setup

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. API Keys Configuration

Create a `.env` file in the project root with the following variables:

```
# Required API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
BRAVE_API_KEY=your_brave_api_key_here
NOTION_TOKEN=your_notion_token_here

# Optional Settings
NOTION_PAGE_ID=your_notion_page_id_here
CHROMA_PERSIST_DIRECTORY=research_db
```

- **ANTHROPIC_API_KEY**: Get from [Anthropic Console](https://console.anthropic.com/)
- **BRAVE_API_KEY**: Get from [Brave Search API](https://api.search.brave.com/)
- **NOTION_TOKEN**: Get from [Notion Integrations](https://www.notion.so/my-integrations)
- **NOTION_PAGE_ID** (optional): ID of the Notion page where research will be saved

### 3. ChromaDB Setup

The application uses ChromaDB for vector storage with semantic search capabilities. The system uses:

- **PersistentClient**: For reliable on-disk storage
- **SentenceTransformer Embeddings**: Using the "all-MiniLM-L6-v2" model for high-quality embeddings
- **Automatic Data Persistence**: Changes are automatically saved to disk

By default, data is stored in a local directory named `research_db`. You can modify this by setting the `CHROMA_PERSIST_DIRECTORY` environment variable.

```bash
# Optional: Install additional ChromaDB dependencies for better performance
pip install chromadb[all]
```

First-time setup may take longer as the sentence transformer model is downloaded.

### 4. MCP Server Setup

You'll need to set up the following MCP servers:

1. **Brave Search MCP Server**:
   ```bash
   npx -y @modelcontextprotocol/server-brave-search
   ```
   This will initialize the server and provide a path to the server script.

2. **Puppeteer MCP Server**:
   ```bash
   npx -y @modelcontextprotocol/server-puppeteer
   ```

3. **Notion MCP Server**:
   ```bash
   npx -y @modelcontextprotocol/server-notion
   ```

Make note of the paths to each server script, as you'll need them when running the application.

## Usage

Run the application with:

```bash
python main.py "your research query" \
    --brave-server path/to/brave/server.js \
    --puppeteer-server path/to/puppeteer/server.js \
    --notion-server path/to/notion/server.js
```

### Example:

```bash
python main.py "The impact of quantum computing on cybersecurity" \
    --brave-server ./node_modules/.bin/mcp-server-brave-search \
    --puppeteer-server ./node_modules/.bin/mcp-server-puppeteer \
    --notion-server ./node_modules/.bin/mcp-server-notion
```

## How It Works

1. **MCP Server Connection**:
   - Connects to each MCP server
   - Lists available tools and their parameters

2. **Anthropic Tool Calling for Search**:
   - Converts MCP tool schemas to Anthropic tool format
   - Uses Claude to intelligently select and call Brave search tools
   - No fallbacks - Claude is fully responsible for tool selection

3. **Anthropic Tool Calling for Scraping**:
   - Analyzes each URL to determine the best scraping approach
   - Uses Claude to select appropriate Puppeteer tools based on the content
   - Strictly follows MCP philosophy - only scrapes URLs where Claude selects a tool

4. **Content Processing**:
   - Stores all data in the ChromaDB vector database with semantic embeddings
   - Generates a comprehensive summary using Claude

5. **Result Organization**:
   - Saves organized research results to Notion
   - Maintains a persistent local database of all research data for future queries

## MCP Implementation Details

The application follows a pure Model Context Protocol (MCP) specification:

1. **Server Discovery**: Connects to MCP servers and discovers available tools
2. **Tool Schema Conversion**: Converts MCP tool schemas to Anthropic tool format
3. **AI-Driven Tool Selection**: Claude has full responsibility for selecting appropriate tools
4. **Tool Execution**: Executes MCP tool calls based on Claude's recommendations without fallbacks

## Advanced Configuration

### ChromaDB Configuration

The application uses ChromaDB's PersistentClient with the sentence-transformers embedding model. You can customize these settings:

- **Storage Path**: Set `CHROMA_PERSIST_DIRECTORY` in your `.env` file
- **Embedding Model**: The code uses "all-MiniLM-L6-v2" by default, but you can modify it for different models
- **Collection Configuration**: Metadata and settings can be adjusted in the code

For production use, consider setting up a dedicated directory on persistent storage.

### Notion Integration

For better Notion integration:

1. Create a dedicated Notion integration at [Notion Integrations](https://www.notion.so/my-integrations)
2. Share a specific page or database with your integration
3. Copy the page ID and set it as `NOTION_PAGE_ID` in your `.env` file

## Troubleshooting

### Connection Issues
- **Problem**: Unable to connect to MCP servers
- **Solution**: Ensure servers are running and paths are correct. Try running the server commands in a separate terminal to verify they work.

### API Key Errors
- **Problem**: "Missing required environment variables" error
- **Solution**: Check that all required API keys are set in your `.env` file with the correct variable names.

### Tool Calling Failures
- **Problem**: Claude is not selecting tools
- **Solution**: Ensure you're using Anthropic API version 0.18.0+ and Claude 3 Sonnet or higher model.

### Missing Data
- **Problem**: URLs aren't being scraped
- **Solution**: Check the logs to see if Claude is selecting tools. If not, try updating the prompts in the code.

### ChromaDB Errors
- **Problem**: "Failed to initialize ChromaDB" error
- **Solution**: Ensure the directory specified in `CHROMA_PERSIST_DIRECTORY` is writable.
- **Problem**: Slow first-time startup
- **Solution**: This is normal as the sentence transformer model downloads. Subsequent runs will be faster.


