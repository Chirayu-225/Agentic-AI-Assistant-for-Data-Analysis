"""Core principle: core/* has NO LLM imports and never touches the network."""
import ast
import pathlib

import pytest

CORE = pathlib.Path(__file__).resolve().parent.parent / "core"
FORBIDDEN = {"groq", "openai", "anthropic", "google", "ollama", "requests", "httpx", "urllib3",
             "aiohttp", "socket", "llm", "agent", "streamlit", "langchain", "sqlite3"}


@pytest.mark.parametrize("path", sorted(CORE.glob("*.py")), ids=lambda p: p.name)
def test_core_module_has_no_llm_or_network_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert not imported & FORBIDDEN, f"{path.name} imports {imported & FORBIDDEN}"
