try:
    from mcp.server.mcpserver import MCPServer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MCP is not installed. Run: pip install -r requirements-mcp.txt") from exc

from backend.mcp.tools import analyze_sentiment, analyze_text

mcp = MCPServer("contextual-sentiment-analyzer")
mcp.tool()(analyze_sentiment)
mcp.tool()(analyze_text)

if __name__ == "__main__":
    mcp.run()
