"""
tools/date_tool.py
Returns current date/time. Used for queries like
'when will TCS visit' — tells user we don't have schedule data.
"""
from datetime import datetime
from tools.base_tool import BaseTool, ToolResult


class DateTool(BaseTool):

    @property
    def name(self) -> str:
        return "current_date"

    @property
    def description(self) -> str:
        return "Returns current date and time."

    def execute(self, query: str) -> ToolResult:
        now = datetime.now()
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                f"Current date: {now.strftime('%A, %d %B %Y')}. "
                f"Time: {now.strftime('%H:%M')} IST. "
                "Note: Campus visit schedules are not available in this dataset. "
                "Please check with your placement cell or official SVECW notices."
            ),
        )