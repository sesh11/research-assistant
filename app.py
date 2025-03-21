import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
import json
import time
from client import MCPClient, ResearchResult
import threading
from queue import Queue
import re
from mcp_flow import add_mcp_flow_visualization


# Configure page
st.set_page_config(
    page_title="MCP Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .server-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .tool-list {
        max-height: 200px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-top: 10px;
    }
    .log-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        font-family: monospace;
    }
    .step-header {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .tool-invocation {
        background-color: #fff8e1;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 3px solid #ffc107;
    }
    .summary-container {
        background-color: #f1f8e9;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #8bc34a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'connected_servers' not in st.session_state:
    st.session_state.connected_servers = {}
if 'server_tools' not in st.session_state:
    st.session_state.server_tools = {}
if 'research_result' not in st.session_state:
    st.session_state.research_result = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'tool_invocations' not in st.session_state:
    st.session_state.tool_invocations = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Custom logger to capture output
class StreamlitLogger:
    def __init__(self, queue):
        self.queue = queue
        
    def write(self, text):
        if text.strip():  # Only process non-empty lines
            self.queue.put(text)
        return len(text)
    
    def flush(self):
        pass

# Function to parse logs and update UI state
def process_logs(log_queue):
    while True:
        if not log_queue.empty():
            log = log_queue.get()
            st.session_state.logs.append(log)
            
            # Parse server connections
            if "Connected to" in log and "server with tools" in log:
                server_name = log.split("Connected to ")[1].split(" server")[0]
                st.session_state.connected_servers[server_name] = True
            
            # Parse tool invocations
            if "Sending" in log and "request to Claude" in log:
                tool_match = re.search(r"Sending (\w+) request to Claude", log)
                if tool_match:
                    tool_name = tool_match.group(1)
                    st.session_state.tool_invocations.append({
                        "tool": tool_name,
                        "timestamp": time.time(),
                        "status": "started"
                    })
            
            # Update progress based on steps
            if "Step " in log:
                step_match = re.search(r"Step (\d+)/(\d+)", log)
                if step_match:
                    current = int(step_match.group(1))
                    total = int(step_match.group(2))
                    st.session_state.current_step = current
                    st.session_state.progress = current / total
            
            # Mark research as complete
            if "Results have been saved" in log:
                st.session_state.is_running = False
                
        time.sleep(0.1)

# Main research function
async def run_research(query, server_paths, log_queue):
    st.session_state.is_running = True
    st.session_state.logs = []
    st.session_state.connected_servers = {}
    st.session_state.server_tools = {}
    st.session_state.tool_invocations = []
    st.session_state.current_step = 0
    st.session_state.progress = 0
    
    # Redirect stdout to our custom logger
    original_stdout = sys.stdout
    sys.stdout = StreamlitLogger(log_queue)
    
    try:
        client = MCPClient()
        
        # Connect to servers
        for server_name, server_path in server_paths.items():
            if server_path:
                await client.connect_to_server(server_name, server_path)
                st.session_state.server_tools[server_name] = client.server_tools.get(server_name, [])
        
        # Run research
        result = await client.research_topic(query)
        st.session_state.research_result = result
        
        # Clean up
        await client.cleanup()
        
    except Exception as e:
        st.session_state.logs.append(f"Error: {str(e)}")
    finally:
        sys.stdout = original_stdout
        st.session_state.is_running = False

# Sidebar for configuration
st.sidebar.title("MCP Research Assistant")
st.sidebar.markdown("Configure your research parameters and server paths")

query = st.sidebar.text_area("Research Query", 
    "Latest advancements in model context protocol and tell me about the latest pull request to update model specifications",
    height=100)

# Server paths
st.sidebar.subheader("Server Paths")
brave_server = st.sidebar.text_input("Brave Server Path", "/path/to/brave-search/dist/index.js")
puppeteer_server = st.sidebar.text_input("Puppeteer Server Path", "/path/to/puppeteer/dist/index.js")
notion_server = st.sidebar.text_input("Notion Server Path", "/path/to/notion-server/build/index.js")
github_server = st.sidebar.text_input("GitHub Server Path", "/path/to/github/dist/index.js")

# Start research button
if st.sidebar.button("Start Research", disabled=st.session_state.is_running):
    # Create a queue for log processing
    log_queue = Queue()
    
    # Start log processing in a separate thread
    log_thread = threading.Thread(target=process_logs, args=(log_queue,), daemon=True)
    log_thread.start()
    
    # Start research in a separate thread
    server_paths = {
        'brave': brave_server,
        'puppeteer': puppeteer_server,
        'notion': notion_server,
        'github': github_server
    }
    
    research_thread = threading.Thread(
        target=lambda: asyncio.run(run_research(query, server_paths, log_queue)),
        daemon=True
    )
    research_thread.start()

# Main content area
st.title("MCP Research Assistant")

# Progress indicator
if st.session_state.is_running:
    st.progress(st.session_state.progress)
    st.write(f"Step {st.session_state.current_step} in progress...")
elif st.session_state.current_step > 0:
    st.progress(1.0)
    st.success("Research completed!")

# Display connected servers and their tools
st.header("MCP Servers and Tools")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Connected Servers")
    if st.session_state.connected_servers:
        for server, connected in st.session_state.connected_servers.items():
            if connected:
                st.markdown(f"""
                <div class="server-card">
                    <h4>✅ {server.capitalize()} Server</h4>
                    <p>Connection established via stdio</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No servers connected yet")

with col2:
    st.subheader("Available Tools")
    if st.session_state.server_tools:
        for server, tools in st.session_state.server_tools.items():
            with st.expander(f"{server.capitalize()} Tools ({len(tools)})"):
                for tool in tools:
                    st.markdown(f"""
                    <div>
                        <strong>{tool.name}</strong>: {tool.description}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No tools discovered yet")

# Tool invocations
st.header("Tool Invocations")
if st.session_state.tool_invocations:
    for idx, invocation in enumerate(st.session_state.tool_invocations):
        st.markdown(f"""
        <div class="tool-invocation">
            <h4>🔧 {invocation['tool'].capitalize()} Tool</h4>
            <p>Invoked at: {time.strftime('%H:%M:%S', time.localtime(invocation['timestamp']))}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No tools have been invoked yet")

# Research results
if st.session_state.research_result:
    st.header("Research Results")
    
    tabs = st.tabs(["Summary", "Web Content", "GitHub Data", "Scraped Data", "Raw Logs"])
    
    with tabs[0]:
        st.markdown(f"""
        <div class="summary-container">
            <h3>Research Summary</h3>
            <p>{st.session_state.research_result.summary}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        if hasattr(st.session_state.research_result, 'web_content') and st.session_state.research_result.web_content:
            for idx, item in enumerate(st.session_state.research_result.web_content):
                with st.expander(f"Web Result {idx+1}"):
                    st.json(item)
        else:
            st.info("No web content collected")
    
    with tabs[2]:
        if hasattr(st.session_state.research_result, 'github_data') and st.session_state.research_result.github_data:
            for idx, item in enumerate(st.session_state.research_result.github_data):
                with st.expander(f"GitHub Result {idx+1}"):
                    st.json(item)
        else:
            st.info("No GitHub data collected")
    
    with tabs[3]:
        if hasattr(st.session_state.research_result, 'scraped_data') and st.session_state.research_result.scraped_data:
            for idx, item in enumerate(st.session_state.research_result.scraped_data):
                with st.expander(f"Scraped Content {idx+1}"):
                    st.json(item)
        else:
            st.info("No scraped data collected")
    
    with tabs[4]:
        st.subheader("Process Logs")
        st.markdown('<div class="log-container">', unsafe_allow_html=True)
        for log in st.session_state.logs:
            st.text(log)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Show logs if research is running or has been run
    if st.session_state.logs:
        st.header("Process Logs")
        st.markdown('<div class="log-container">', unsafe_allow_html=True)
        for log in st.session_state.logs:
            st.text(log)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Enter a research query and start the process to see results")

# Add this before the footer
add_mcp_flow_visualization()

# Footer
st.markdown("---")
st.markdown("Model Context Protocol (MCP) Research Assistant | Built with Streamlit")