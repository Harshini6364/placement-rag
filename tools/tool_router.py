"""
tools/tool_router.py
Tool router — classifies query and dispatches to correct tool.
Acts as a simple agent: decide → act → observe → respond.
Interface Segregation: each tool is independent.
"""
import re
import logging
from dataclasses import dataclass
from tools.base_tool import BaseTool, ToolResult
from tools.web_search_tool import WebSearchTool
from tools.calculator_tool import CalculatorTool
from tools.date_tool import DateTool
from tools.opinion_guard_tool import OpinionGuardTool

logger = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    needs_tool: bool
    tool_name: str
    reason: str
    result: ToolResult | None = None


# Signals that indicate a query needs a tool instead of / in addition to RAG
TOOL_SIGNALS = {
    "web_search": [
        "campus visit date", "when will", "visit svecw", "schedule",
        "stock price", "current price", "market cap", "share price",
        "work from home", "wfh", "remote work", "work mode",
        "how many students placed", "institution-specific",
        "latest news", "recent", "today",
    ],
    "calculator": [
        "ratio", "package-to-cgpa", "best ratio",
        "calculate", "compute", "how much",
        "cgpa 5", "cgpa 6", "cgpa 7", "cgpa 8", "cgpa 9",
        "cgpa of 5", "cgpa of 6", "cgpa of 7",      
        "cgpa 5.0", "cgpa 4.0", "cgpa 3.0",
        "highest in the world",    
        "pays the most in the world",            
        "with backlog", "i have cgpa", "my cgpa is",  
        "where can i apply", "can i apply",
        "eligible", "qualify",
    ],
    "current_date": [
        "what date", "today", "current date", "what day",
        "when is", "what time",
    ],
    "opinion_guard": [
        "should i join", "which is better", "better career",
        "recommend", "suggest", "what do you think",
        "google or microsoft", "amazon or google",
        "tcs or infosys", "which company should",
    ],
}


class ToolRouter:
    """
    Classifies queries and routes to appropriate tools.
    Falls back gracefully when no tool matches.
    """

    def __init__(self):
        self.tools: dict[str, BaseTool] = {
            "web_search":    WebSearchTool(),
            "calculator":    CalculatorTool(),
            "current_date":  DateTool(),
            "opinion_guard": OpinionGuardTool(),
        }

    def route(self, query: str) -> RouterDecision:
        """
        Returns a RouterDecision.
        If needs_tool=False, RAG pipeline handles it normally.
        """
        q = query.lower()

        for tool_name, signals in TOOL_SIGNALS.items():
            for signal in signals:
                if signal in q:
                    logger.info(f"Tool router: '{signal}' → {tool_name}")
                    result = self.tools[tool_name].execute(query)
                    return RouterDecision(
                        needs_tool=True,
                        tool_name=tool_name,
                        reason=f"Matched signal: '{signal}'",
                        result=result,
                    )

        # Below all CGPA thresholds
        cgpa_match = re.search(r'cgpa\s*(?:of\s*)?(\d+\.?\d*)', q)
        if cgpa_match:
            cgpa = float(cgpa_match.group(1))
            if cgpa < 6.3:
                result = ToolResult(
                    tool_name="calculator",
                    success=True,
                    output=(
                        f"No company in this dataset accepts CGPA {cgpa}. "
                        "The lowest cutoff is Samsung R&D at 6.3. "
                        "Consider improving your CGPA or looking at other opportunities."
                    ),
                )
                return RouterDecision(
                    needs_tool=True,
                    tool_name="calculator",
                    reason=f"CGPA {cgpa} below all thresholds",
                    result=result,
                )

        return RouterDecision(
            needs_tool=False,
            tool_name="rag",
            reason="In-corpus query — handled by RAG pipeline",
        )

    def get_tool_info(self) -> list[dict]:
        """Returns info about all available tools — for UI display."""
        return [
            {"name": t.name, "description": t.description}
            for t in self.tools.values()
        ]