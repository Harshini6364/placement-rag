"""
tools/base_tool.py
Abstract base for all tools. Every tool has a name, description,
and an execute method. SOLID: Open/Closed — add tools without
touching the router.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    source_url: str = ""
    error: str = ""


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def execute(self, query: str) -> ToolResult:
        ...