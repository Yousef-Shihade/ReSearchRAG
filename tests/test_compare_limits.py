
import sys
import os
import re
from unittest.mock import MagicMock

# Mock streamlit before importing app
sys.modules["streamlit"] = MagicMock()
sys.modules["streamlit"].cache_resource = lambda func: func
sys.modules["streamlit"].session_state = {}

# Mock os.getenv to return a fake key for GROQ_API_KEY
original_getenv = os.getenv
def mock_getenv(key, default=None):
    if key == "GROQ_API_KEY":
        return "fake_key"
    return original_getenv(key, default)
os.getenv = mock_getenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import app
except ImportError as e:
    print(f"Failed to import app: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error during app import: {e}")
    sys.exit(1)

def test_resolve_papers_limits():
    print("Testing resolve_papers_from_query limits...")
    
    # Mock build_library_catalog
    # Title is "Attention Is All You Need" (normalized: "attention is all you need")
    app.build_library_catalog = MagicMock(return_value={
        "title_index": [("attention is all you need", "paper1"), ("bert: pre-training of deep bidirectional transformers", "paper2")]
    })
    
    app.st.session_state = {
        "uploaded_number_map": {},
        "uploaded_papers": {}
    }

    # Case 1: Partial title (prefix)
    q = "compare attention vs bert"
    ids = app.resolve_papers_from_query(q)
    print(f"Query: '{q}' -> IDs: {ids}")
    # Expected: Empty or fails to find paper1 if full title is required
    if "paper1" in ids:
        print("  - Found paper1 via partial title!")
    else:
        print("  - Did NOT find paper1 via partial title.")

    # Case 2: Full title
    q = "compare attention is all you need vs bert: pre-training of deep bidirectional transformers"
    ids = app.resolve_papers_from_query(q)
    print(f"Query: '{q}' -> IDs: {ids}")
    if "paper1" in ids:
        print("  - Found paper1 via full title.")
    else:
        print("  - Did NOT find paper1 via full title.")

    # Case 3: Case insensitive
    q = "compare Attention Is All You Need vs BERT"
    # Note: "BERT" is partial for "bert: pre-training..." so strictly it shouldn't match if full match required
    # But "Attention Is All You Need" is full match (ignoring case)
    ids = app.resolve_papers_from_query(q)
    print(f"Query: '{q}' -> IDs: {ids}")
    
    # Case 4: ID with space
    q = "compare paper 1 and paper 2"
    ids = app.resolve_papers_from_query(q)
    print(f"Query: '{q}' -> IDs: {ids}")
    if "paper1" in ids:
        print("  - Found paper1 via 'paper 1'")

    print("test_resolve_papers_limits finished")

if __name__ == "__main__":
    test_resolve_papers_limits()
