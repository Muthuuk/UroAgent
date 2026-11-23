"""Run the UroAgent clinical tools as a real Model Context Protocol (MCP) server.

    pip install mcp
    python -m uroagent.mcp_server          # stdio transport, for MCP clients such as Claude Desktop

Every tool in uroagent.tools.TOOLS is registered. All data are synthetic.
"""
from .tools import TOOLS


def build_server():
    try:                                    # MCP Python SDK 2.x
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:                     # MCP Python SDK 1.x
        from mcp.server.fastmcp import FastMCP as Server
    server = Server("uroagent-clinical-tools")
    for name, fn in TOOLS.items():
        server.tool(name=name, description=(fn.__doc__ or "").strip())(fn)
    return server


if __name__ == "__main__":
    build_server().run()
