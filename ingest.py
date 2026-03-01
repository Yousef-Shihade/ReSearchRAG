"""
build_library_index.py

This script builds the *library* FAISS index for ReSearchRAG.

Overview
--------
ReSearchRAG supports two retrieval scopes:
1) A pre-built on-disk library index (this script builds it).
2) Per-session uploaded documents indexed in-memory (handled in app.py).

This script:
- Loads all PDF files from DATA_DIR
- Extracts text into LangChain Documents with stable metadata
- Splits documents into overlapping chunks suitable for semantic retrieval
- Embeds chunks using a sentence-transformer embedding model
- Builds a FAISS vector index and saves it under INDEX_DIR

Important Design Choice
-----------------------
We assign a *stable* identifier per paper:
    paper_id = <filename without ".pdf">

The Streamlit app (app.py) relies on this stable identity to:
- Display a catalog (paper1/paper2/... mapping)
- Filter retrieved chunks by the correct paper
- Show user-friendly titles in the sidebar

If you rename files, paper_id will change accordingly.
"""



import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

from langchain_community.document_loaders.pdf import PyPDFLoader

# Directory containing the PDF papers to index (library corpus)
DATA_DIR = "data/papers"

# Directory where the FAISS index will be saved on disk
INDEX_DIR = "indexes/faiss_index"


def _first_reasonable_title(text: str, fallback: str) -> str:
    """
    Heuristic title extraction.

    We try to derive a human-readable paper title from the first page text,
    which is useful for displaying a library catalog in the Streamlit UI.

    Strategy:
    - Return the first "reasonable" line whose length is in [8, 180]
    - Otherwise, return a trimmed preview of the text or the fallback string
    """
    if not text:
        return fallback
    for line in text.splitlines():
        line = line.strip()
        if 8 <= len(line) <= 180:
            return line
    t = text.strip().replace("\n", " ")
    return (t[:180] + "...") if len(t) > 180 else (t or fallback)


def load_pdfs(data_dir: str):
    """
    Load all PDFs in `data_dir` into a list[Document].

    Each returned Document represents one PDF page (depending on the loader),
    and is enriched with metadata used later during retrieval and UI display.

    Metadata conventions (critical):
    - paper_id: stable paper identifier used for filtering and catalog grouping
    - title: best-effort human-readable title (used in sidebar/catalog)
    - source: original filename (useful for debugging / traceability)
    - page: page number (if available)
    """
    docs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(data_dir, fname)
            print(f"✓ Loading {path}")

            loader = None
            if HAVE_PYMUPDF:
                try:
                    loader = PyMuPDFLoader(path)
                except Exception:
                    loader = None
            if loader is None:
                loader = PyPDFLoader(path)

            file_docs = loader.load()

            paper_name = os.path.splitext(fname)[0]  # stable id (filename without .pdf)

            # best-effort title from first page
            first_text = file_docs[0].page_content.strip() if file_docs else ""
            title = _first_reasonable_title(first_text, fallback=paper_name)

            for d in file_docs:
                # Stable identity (app.py relies on this)
                d.metadata["paper_id"] = paper_name
                d.metadata["paper_name"] = paper_name

                # Optional: store title to show in sidebar
                d.metadata["title"] = title

                # keep source as filename
                d.metadata["source"] = fname

                # Ensure page exists if loader didn’t set it
                if "page" not in d.metadata:
                    d.metadata["page"] = d.metadata.get("page", 0)

            docs.extend(file_docs)

    return docs


def main():
    """
    Entry point.

    Steps:
    1) Load environment variables (.env), mainly for consistency with the project setup
    2) Load PDFs from DATA_DIR
    3) Split documents into overlapping chunks for semantic retrieval
    4) Embed chunks with a sentence-transformer model
    5) Build and save FAISS index to INDEX_DIR
    """
    
    load_dotenv()
    docs = load_pdfs(DATA_DIR)
    if not docs:
        print("No PDFs found in data/papers. Add some research papers and try again.")
        return

    # Larger chunks helps keep "Results + discussion + table captions" together more often
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")

    if HAVE_PYMUPDF:
        print("Note: Used PyMuPDFLoader (when available) for PDF extraction.")
    else:
        print("Note: PyMuPDFLoader not available; used PyPDFLoader.")


if __name__ == "__main__":
    main()
