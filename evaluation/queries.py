"""
evaluation/queries.py
All 30 official evaluation queries from Section 9 of the PDF.
"""

EVAL_QUERIES = [
    # Easy
    {"id": "E1", "query": "What is the CGPA requirement for TCS?", "difficulty": "Easy", "expected_keywords": ["7.5"]},
    {"id": "E2", "query": "How many backlogs does Deloitte allow?", "difficulty": "Easy", "expected_keywords": ["1"]},
    {"id": "E3", "query": "What is the bond period for Amazon?", "difficulty": "Easy", "expected_keywords": ["2"]},
    {"id": "E4", "query": "Which technology does Flipkart focus on in interviews?", "difficulty": "Easy", "expected_keywords": ["Python"]},
    {"id": "E5", "query": "What is the package offered by Google?", "difficulty": "Easy", "expected_keywords": ["42.0", "42"]},
    {"id": "E6", "query": "Does Microsoft allow backlogs?", "difficulty": "Easy", "expected_keywords": ["Yes", "1"]},
    {"id": "E7", "query": "What rounds does TCS conduct?", "difficulty": "Easy", "expected_keywords": ["Round 1", "Round 2", "Round 3"]},
    {"id": "E8", "query": "Which programming language is tested at Amazon?", "difficulty": "Easy", "expected_keywords": ["C++"]},
    # Medium
    {"id": "M1", "query": "List all companies that allow at least 2 backlogs.", "difficulty": "Medium", "expected_keywords": ["Flipkart", "IBM", "Qualcomm", "Samsung"]},
    {"id": "M2", "query": "Which companies require a CGPA above 8.0?", "difficulty": "Medium", "expected_keywords": ["Infosys", "Accenture", "Cognizant", "SAP", "HCL"]},
    {"id": "M3", "query": "Which company has the highest package among IT service firms?", "difficulty": "Medium", "expected_keywords": ["Infosys", "42.9"]},
    {"id": "M4", "query": "Which companies are bond-free?", "difficulty": "Medium", "expected_keywords": ["TCS", "Infosys", "Microsoft", "IBM", "Intel"]},
    {"id": "M5", "query": "Compare TCS and Infosys on all eligibility criteria.", "difficulty": "Medium", "expected_keywords": ["TCS", "Infosys", "CGPA", "package"]},
    {"id": "M6", "query": "How many SDE roles does Amazon hire versus Google?", "difficulty": "Medium", "expected_keywords": ["42", "30"]},
    {"id": "M7", "query": "Which company hires the most Interns?", "difficulty": "Medium", "expected_keywords": ["Oracle", "95"]},
    {"id": "M8", "query": "What topics should I prepare for a Microsoft interview?", "difficulty": "Medium", "expected_keywords": ["DSA", "OS", "DBMS"]},
    {"id": "M9", "query": "Which company's package grew the most from 2021 to 2024?", "difficulty": "Medium", "expected_keywords": ["Infosys", "6.9"]},
    {"id": "M10", "query": "Which companies use Python as the technical focus?", "difficulty": "Medium", "expected_keywords": ["Google", "Flipkart", "Oracle", "Intel"]},
    # Hard
    {"id": "H1", "query": "A student with CGPA 7.0, 1 backlog wants maximum pay with no bond.", "difficulty": "Hard", "expected_keywords": ["Wipro", "26.1"]},
    {"id": "H2", "query": "Which Python-focused company hires the most Interns?", "difficulty": "Hard", "expected_keywords": ["Oracle", "95"]},
    {"id": "H3", "query": "For CGPA 8.0+, zero backlog students, rank companies by package.", "difficulty": "Hard", "expected_keywords": ["Infosys", "Cognizant", "Capgemini"]},
    {"id": "H4", "query": "Which company had conflicting CGPA data across sources?", "difficulty": "Hard", "expected_keywords": ["TCS", "Amazon", "Google", "Infosys", "Microsoft"]},
    {"id": "H5", "query": "Is the Amazon CGPA cutoff 6.4 or 7.0? Explain.", "difficulty": "Hard", "expected_keywords": ["6.4", "7.0", "conflict", "official", "portal"]},
    {"id": "H6", "query": "Which company offers the best package-to-CGPA ratio?", "difficulty": "Hard", "expected_keywords": ["Infosys", "Samsung"]},
    {"id": "H7", "query": "Compare Google and Amazon on all dimensions: eligibility, package, hiring, trend.", "difficulty": "Hard", "expected_keywords": ["Google", "Amazon", "42.0", "28.6"]},
    # Expert (fallback)
    {"id": "X1", "query": "What is TCS's campus visit date at SVECW?", "difficulty": "Expert", "expected_fallback": True},
    {"id": "X2", "query": "Should I join Google or Microsoft? Which is better for my career?", "difficulty": "Expert", "expected_fallback": True},
    {"id": "X3", "query": "I have CGPA 5.0. Where can I apply?", "difficulty": "Expert", "expected_keywords": ["no company", "5.0", "cutoff"]},
    {"id": "X4", "query": "What is Infosys's current stock price?", "difficulty": "Expert", "expected_fallback": True},
    {"id": "X5", "query": "Which company in this dataset pays the highest in the world?", "difficulty": "Expert", "expected_fallback": True},
]