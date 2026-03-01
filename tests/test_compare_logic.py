
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

def test_is_compare():
    print("Testing is_compare...")
    assert app.is_compare("compare paper1 vs paper2"), "Failed: compare paper1 vs paper2"
    assert app.is_compare("what is the difference between A and B"), "Failed: difference"
    assert app.is_compare("paper1 versus paper2"), "Failed: versus"
    assert not app.is_compare("summary of paper1"), "Failed: summary should be False"
    print("test_is_compare passed")

def test_resolve_papers():
    print("Testing resolve_papers_from_query...")
    
    # Mock build_library_catalog
    # It returns a dict with "title_index": [(norm_title, key), ...]
    app.build_library_catalog = MagicMock(return_value={
        "title_index": [("paper title a", "paper1"), ("paper title b", "paper2")]
    })
    
    # Mock st.session_state for uploaded papers
    app.st.session_state = {
        "uploaded_number_map": {"upload1": "uid1"},
        "uploaded_papers": {
            "uid1": {"title": "Uploaded Paper Title"}
        }
    }

    # Case 1: ID references
    q = "compare paper1 and paper2"
    ids = app.resolve_papers_from_query(q)
    print(f"Query: '{q}' -> IDs: {ids}")
    assert "paper1" in ids and "paper2" in ids

    # Case 2: Title references
    q = "compare paper title a vs paper title b"
    ids = app.resolve_papers_from_query(q)
    print(f"Query: '{q}' -> IDs: {ids}")
    assert "paper1" in ids and "paper2" in ids

    # Case 3: Mixed ID and Title
    q = "difference between paper1 and paper title b"
    ids = app.resolve_papers_from_query(q)
    print(f"Query: '{q}' -> IDs: {ids}")
    assert "paper1" in ids and "paper2" in ids
    
    # Case 4: Upload reference
    q = "compare upload1 vs paper1"
    ids = app.resolve_papers_from_query(q)
    print(f"Query: '{q}' -> IDs: {ids}")
    assert "upload1" in ids and "paper1" in ids

    print("test_resolve_papers passed")

if __name__ == "__main__":
    test_is_compare()
    test_resolve_papers()
