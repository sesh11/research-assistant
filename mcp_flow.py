import streamlit as st
import graphviz

def render_mcp_flow():
    """Render a visualization of the MCP flow process"""
    
    # Create a graphviz object
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB', size='8,5')
    
    # Add nodes
    graph.node('client', 'MCP Client', shape='box', style='filled', fillcolor='lightblue')
    graph.node('brave', 'Brave Search Server', shape='box', style='filled', fillcolor='lightgreen')
    graph.node('puppeteer', 'Puppeteer Server', shape='box', style='filled', fillcolor='lightgreen')
    graph.node('notion', 'Notion Server', shape='box', style='filled', fillcolor='lightgreen')
    graph.node('github', 'GitHub Server', shape='box', style='filled', fillcolor='lightgreen')
    graph.node('claude', 'Claude API', shape='box', style='filled', fillcolor='lightyellow')
    graph.node('chromadb', 'ChromaDB', shape='cylinder', style='filled', fillcolor='lightgrey')
    
    # Add edges
    graph.edge('client', 'brave', label='1. Connect (stdio)')
    graph.edge('client', 'puppeteer', label='2. Connect (stdio)')
    graph.edge('client', 'notion', label='3. Connect (stdio)')
    graph.edge('client', 'github', label='4. Connect (stdio)')
    
    graph.edge('brave', 'client', label='5. Return tools')
    graph.edge('puppeteer', 'client', label='6. Return tools')
    graph.edge('notion', 'client', label='7. Return tools')
    graph.edge('github', 'client', label='8. Return tools')
    
    graph.edge('client', 'claude', label='9. Send query with tools')
    graph.edge('claude', 'client', label='10. Select tools to use')
    
    graph.edge('client', 'brave', label='11. Execute search')
    graph.edge('client', 'github', label='12. Execute search')
    graph.edge('client', 'puppeteer', label='13. Scrape content')
    
    graph.edge('client', 'claude', label='14. Generate summary')
    graph.edge('client', 'notion', label='15. Save results')
    graph.edge('client', 'chromadb', label='16. Store data')
    
    return graph

def add_mcp_flow_visualization():
    """Add MCP flow visualization to the Streamlit app"""
    
    st.header("MCP Flow Visualization")
    
    with st.expander("View MCP Process Flow", expanded=True):
        st.graphviz_chart(render_mcp_flow())
        
        st.markdown("""
        ### MCP Process Explained
        
        1. **Connection Phase**: The client establishes stdio connections with each MCP server
        2. **Tool Discovery**: Each server reports available tools to the client
        3. **Planning Phase**: Claude analyzes the query and decides which tools to use
        4. **Execution Phase**: The client invokes selected tools on appropriate servers
        5. **Synthesis Phase**: Results are collected and summarized by Claude
        6. **Storage Phase**: Results are saved to Notion and ChromaDB
        
        This process demonstrates how MCP enables AI systems to interact with external tools and data sources in a structured way.
        """)