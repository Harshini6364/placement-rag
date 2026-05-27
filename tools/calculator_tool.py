"""
tools/calculator_tool.py
Safe calculator for placement math:
package-to-CGPA ratio, filtering, growth calculations.
"""
import re
import logging
from tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Placement data for calculations
PLACEMENT_DATA = {
    "TCS":           {"package": 4.1,  "min_cgpa": 7.5, "bond": 0, "backlogs": 0},
    "Infosys":       {"package": 42.9, "min_cgpa": 8.0, "bond": 0, "backlogs": 0},
    "Deloitte":      {"package": 9.6,  "min_cgpa": 7.7, "bond": 1, "backlogs": 1},
    "Accenture":     {"package": 17.3, "min_cgpa": 8.2, "bond": 2, "backlogs": 0},
    "Amazon":        {"package": 28.6, "min_cgpa": 6.4, "bond": 2, "backlogs": 1},
    "Flipkart":      {"package": 25.3, "min_cgpa": 7.8, "bond": 2, "backlogs": 2},
    "Google":        {"package": 42.0, "min_cgpa": 7.4, "bond": 1, "backlogs": 0},
    "Microsoft":     {"package": 21.4, "min_cgpa": 6.1, "bond": 0, "backlogs": 1},
    "Wipro":         {"package": 26.1, "min_cgpa": 6.7, "bond": 1, "backlogs": 1},
    "Cognizant":     {"package": 42.3, "min_cgpa": 8.4, "bond": 2, "backlogs": 0},
    "Capgemini":     {"package": 38.3, "min_cgpa": 7.1, "bond": 2, "backlogs": 0},
    "IBM":           {"package": 27.5, "min_cgpa": 7.5, "bond": 0, "backlogs": 2},
    "Adobe":         {"package": 18.3, "min_cgpa": 7.5, "bond": 1, "backlogs": 0},
    "Oracle":        {"package": 17.3, "min_cgpa": 7.7, "bond": 2, "backlogs": 0},
    "SAP":           {"package": 20.7, "min_cgpa": 8.4, "bond": 2, "backlogs": 0},
    "HCL":           {"package": 28.1, "min_cgpa": 8.4, "bond": 2, "backlogs": 1},
    "Tech Mahindra": {"package": 35.9, "min_cgpa": 8.1, "bond": 1, "backlogs": 2},
    "Qualcomm":      {"package": 41.3, "min_cgpa": 7.2, "bond": 1, "backlogs": 2},
    "Intel":         {"package": 41.4, "min_cgpa": 7.0, "bond": 0, "backlogs": 0},
    "Samsung R&D":   {"package": 7.6,  "min_cgpa": 6.3, "bond": 2, "backlogs": 2},
}


class CalculatorTool(BaseTool):

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Compute package-to-CGPA ratios, filter eligible companies, rank by package."

    def execute(self, query: str) -> ToolResult:
        q = query.lower()
        try:
            if "ratio" in q or "package-to-cgpa" in q or "best ratio" in q:
                return self._package_cgpa_ratio()

            if "cgpa" in q and any(w in q for w in ["apply", "eligible", "qualify", "backlog"]):
                return self._eligibility_filter(query)
            
            if "world" in q or "globally" in q:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=(
                        "This dataset covers only the listed companies for SVECW placements. "
                        "Among listed companies, Infosys offers the highest package at 42.9 LPA. "
                        "For global rankings, this dataset does not have that scope."
                    ),
                )

            if any(w in q for w in ["highest", "maximum", "best package", "most pay"]):
                return self._rank_by_package()

            # Generic math expression
            return self._eval_expression(query)

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Calculation failed.",
                error=str(e),
            )

    def _package_cgpa_ratio(self) -> ToolResult:
        ratios = {
            company: round(data["package"] / data["min_cgpa"], 3)
            for company, data in PLACEMENT_DATA.items()
        }
        ranked = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        lines = [f"{i+1}. {c}: {r} LPA/CGPA" for i, (c, r) in enumerate(ranked[:5])]
        return ToolResult(
            tool_name=self.name,
            success=True,
            output="Top 5 package-to-CGPA ratio:\n" + "\n".join(lines),
        )

    def _eligibility_filter(self, query: str) -> ToolResult:
        # Extract CGPA and backlog from query
        cgpa_match = re.search(r'cgpa\s*(?:of\s*|=\s*|:?\s*)(\d+\.?\d*)', query.lower())
        backlog_match = re.search(r'(\d+)\s*backlog', query.lower())

        student_cgpa = float(cgpa_match.group(1)) if cgpa_match else 0.0
        student_backlogs = int(backlog_match.group(1)) if backlog_match else 0

        if student_cgpa == 0:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Could not extract CGPA from query.",
                error="No CGPA found",
            )

        eligible = [
            (c, d) for c, d in PLACEMENT_DATA.items()
            if d["min_cgpa"] <= student_cgpa and d["backlogs"] >= student_backlogs
        ]

        if not eligible:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"No companies found for CGPA {student_cgpa} with {student_backlogs} backlogs.",
            )

        ranked = sorted(eligible, key=lambda x: x[1]["package"], reverse=True)
        lines = [
            f"{i+1}. {c}: {d['package']} LPA (min CGPA {d['min_cgpa']}, bond {d['bond']}yr)"
            for i, (c, d) in enumerate(ranked[:10])
        ]
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                f"Eligible companies for CGPA {student_cgpa}, "
                f"{student_backlogs} backlogs (ranked by package):\n"
                + "\n".join(lines)
            ),
        )

    def _rank_by_package(self) -> ToolResult:
        ranked = sorted(
            PLACEMENT_DATA.items(), key=lambda x: x[1]["package"], reverse=True
        )
        lines = [
            f"{i+1}. {c}: {d['package']} LPA"
            for i, (c, d) in enumerate(ranked[:10])
        ]
        return ToolResult(
            tool_name=self.name,
            success=True,
            output="Companies ranked by package:\n" + "\n".join(lines),
        )

    def _eval_expression(self, query: str) -> ToolResult:
        # Extract and safely evaluate simple math expressions
        expr_match = re.search(r'[\d\s\+\-\*\/\(\)\.]+', query)
        if not expr_match:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="No mathematical expression found.",
                error="No expression",
            )
        expr = expr_match.group(0).strip()
        try:
            result = eval(expr, {"__builtins__": {}})
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f"{expr} = {result}",
            )
        except Exception:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Could not evaluate expression.",
                error="Eval failed",
            )