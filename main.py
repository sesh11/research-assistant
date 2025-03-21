import asyncio
import argparse
import sys
from pathlib import Path
import websockets
import json
from client import MCPClient  # Ensure this import is present

async def websocket_handler(websocket, client):
    try:
        print(f"WebSocket client connected from {websocket.remote_address}, starting research")
        await client.run_research(websocket, client.args.query)
    except websockets.ConnectionClosed:
        print("WebSocket client disconnected")
    finally:
        await client.cleanup()

async def setup_client(args):
    client = MCPClient()
    try:
        await client.connect_to_server('brave', args.brave_server)
        await client.connect_to_server('puppeteer', args.puppeteer_server)
        await client.connect_to_server('notion', args.notion_server)
        await client.connect_to_server('github', args.github_server)
        return client
    except Exception as e:
        print(f"Error setting up client: {str(e)}")
        await client.cleanup()
        sys.exit(1)

async def main_async(args):
    client = await setup_client(args)
    client.args = args
    async with websockets.serve(lambda ws: websocket_handler(ws, client), "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        print("Waiting for a WebSocket client (e.g., index.html) to connect...")
        await asyncio.Future()  # Runs indefinitely until interrupted

def main():
    parser = argparse.ArgumentParser(description='Research Assistant')
    parser.add_argument('query', help='Research topic or question')
    parser.add_argument('--brave-server', required=True, help='Path to Brave MCP server script')
    parser.add_argument('--puppeteer-server', required=True, help='Path to Puppeteer MCP server script')
    parser.add_argument('--notion-server', required=True, help='Path to Notion MCP server script')
    parser.add_argument('--github-server', required=True, help='Path to GitHub MCP server script')
    args = parser.parse_args()

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