"""
version.py
Project metadata.
"""

__version__ = "1.0.0"
__author__ = "SVECW — Department of Information Technology"
__hackathon__ = "RAG-ATHON 24"
__project__ = "Placement Intelligence Assistant"
__description__ = (
    "A production-grade Multimodal RAG system for placement intelligence "
    "with hybrid search, vision chart support, conflict detection, "
    "tool-augmented agent, and 30-query evaluation pipeline."
)


def print_banner():
    print(f"""
╔══════════════════════════════════════════════════════╗
║   {__project__}                                      ║
║   {__hackathon__} · {__author__}                     ║
║   Version: {__version__}                             ║
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_banner()