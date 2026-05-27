"""
tools/opinion_guard_tool.py
Handles subjective/opinion queries gracefully.
'Should I join Google or Microsoft?' → objective comparison, no opinion.
"""
from tools.base_tool import BaseTool, ToolResult

COMPARISON_DATA = {
    ("google", "microsoft"): {
        "Google":    {"package": 42.0, "cgpa": 7.4, "bond": "1yr", "focus": "Python/Algorithms"},
        "Microsoft": {"package": 21.4, "cgpa": 6.1, "bond": "None", "focus": "C++/OS/DBMS"},
        "verdict": "Google pays more (42 vs 21.4 LPA) but has stricter algorithmic interviews. Microsoft has no bond and lower CGPA cutoff. Choice depends on your profile and interview prep.",
    },
    ("amazon", "google"): {
        "Amazon":  {"package": 28.6, "cgpa": 6.4, "bond": "2yr", "focus": "DSA/LLD/C++"},
        "Google":  {"package": 42.0, "cgpa": 7.4, "bond": "1yr", "focus": "Python/Algorithms"},
        "verdict": "Google pays more. Amazon has lower CGPA cutoff (6.4 vs 7.4). Amazon focuses on Leadership Principles in addition to DSA.",
    },
    ("tcs", "infosys"): {
        "TCS":     {"package": 4.1,  "cgpa": 7.5, "bond": "None", "focus": "System Design"},
        "Infosys": {"package": 42.9, "cgpa": 8.0, "bond": "None", "focus": "Java/OOPs"},
        "verdict": "Infosys pays significantly more (42.9 vs 4.1 LPA) but needs higher CGPA (8.0 vs 7.5). Both are bond-free.",
    },
}


class OpinionGuardTool(BaseTool):

    @property
    def name(self) -> str:
        return "opinion_guard"

    @property
    def description(self) -> str:
        return "Handles subjective career questions with objective data comparison."

    def execute(self, query: str) -> ToolResult:
        q = query.lower()

        # Find matching comparison
        for pair, data in COMPARISON_DATA.items():
            if all(company in q for company in pair):
                lines = []
                for company, metrics in data.items():
                    if company == "verdict":
                        continue
                    lines.append(
                        f"{company}: {metrics['package']} LPA | "
                        f"min CGPA {metrics['cgpa']} | "
                        f"bond {metrics['bond']} | "
                        f"focus {metrics['focus']}"
                    )
                lines.append(f"\nObjective summary: {data['verdict']}")
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output="\n".join(lines),
                )

        # Generic subjective query
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                "This is a subjective question — the best choice depends on your "
                "individual profile, CGPA, backlog status, and career goals. "
                "I can provide an objective comparison of any two companies if you name them. "
                "For personal career advice, consult your placement officer."
            ),
        )