import asyncio
import argparse
import sys
from pathlib import Path

from client import MCPClient

async def setup_client(args):
    """Set up and connect to all required MCP servers"""
    client = MCPClient()
    
    try:
        # Connect to Brave MCP server
        await client.connect_to_server('brave', args.brave_server)
        
        # Connect to Puppeteer MCP server
        await client.connect_to_server('puppeteer', args.puppeteer_server)
        
        # Connect to Notion MCP server
        await client.connect_to_server('notion', args.notion_server)
        
        # Connect to GitHub MCP server
        await client.connect_to_server('github', args.github_server)
        
        return client
    except Exception as e:
        print(f"Error setting up client: {str(e)}")
        await client.cleanup()
        sys.exit(1)

async def main_async(args):
    """Async main function that handles the research workflow"""
    client = await setup_client(args)
    
    try:
        print("\nResearching topic:", args.query)
        research_results = await client.research_topic(args.query)
        
        print("\nResearch Summary:")
        print(research_results.summary)
        
        # Display GitHub findings if available
        if research_results.github_data:
            print("\nGitHub Repositories and PRs:")
            for repo_data in research_results.github_data:
                repo = repo_data.get('repository', {})
                prs = repo_data.get('pull_requests', [])
                
                print(f"\n- {repo.get('full_name', 'Unknown repository')}")
                print(f"  Description: {repo.get('description', 'No description')}")
                print(f"  Stars: {repo.get('stargazers_count', 0)}")
                
                if prs:
                    print("  Recent PRs:")
                    for pr in prs[:3]:  # Show top 3 PRs
                        print(f"  - #{pr.get('number', '?')}: {pr.get('title', 'Untitled PR')}")
        
        print("\nResults have been saved to Notion and stored in the local research database.")
        
    except Exception as e:
        print(f"Error during research: {str(e)}")
        raise
    finally:
        await client.cleanup()

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Unified Research Assistant')
    
    # Required arguments
    parser.add_argument('query', help='Research topic or question')
    parser.add_argument('--brave-server', required=True, help='Path to Brave MCP server script')
    parser.add_argument('--puppeteer-server', required=True, help='Path to Puppeteer MCP server script')
    parser.add_argument('--notion-server', required=True, help='Path to Notion MCP server script')
    parser.add_argument('--github-server', required=True, help='Path to GitHub MCP server script')
    
    args = parser.parse_args()
    
    # Validate server paths
    for server_path in [args.brave_server, args.puppeteer_server, args.notion_server, args.github_server]:
        if not Path(server_path).exists():
            print(f"Error: Server script not found: {server_path}")
            sys.exit(1)
    
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
