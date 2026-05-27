"""
tools/web_search_tool.py
Web search using DuckDuckGo (free, no API key needed).
Used for: campus visit dates, current events, real-time info.
"""
import logging
from tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information not in the placement documents."

    def execute(self, query: str) -> ToolResult:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            if not results:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output="No results found.",
                    error="Empty results",
                )
            # Combine top results
            snippets = []
            urls = []
            for r in results:
                snippets.append(f"- {r.get('title', '')}: {r.get('body', '')[:200]}")
                urls.append(r.get("href", ""))

            return ToolResult(
                tool_name=self.name,
                success=True,
                output="\n".join(snippets),
                source_url=urls[0] if urls else "",
            )
        except ImportError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Web search unavailable. Install: pip install duckduckgo-search",
                error="Import error",
            )
        except Exception as e:
            logger.error(f"WebSearchTool error: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"Search failed: {e}",
                error=str(e),
            )