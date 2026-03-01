"""
app.py — ReSearchRAG (Streamlit Application)


Overview
--------
ReSearchRAG is an interactive research assistant that answers questions grounded in a
collection of academic papers. The system supports two document sources:

1) Library papers (persistent):
   - A pre-built FAISS index stored on disk (INDEX_DIR).
   - Created offline using an ingestion script.

2) Uploaded papers (per session):
   - Users can upload multiple PDF/TXT files during a session.
   - Each uploaded file is chunked and indexed locally in memory.

Core requirements implemented
-----------------------------
- Retrieval-Augmented Generation (RAG) with strict grounding:
  The assistant answers ONLY using retrieved context.
  If the answer is not supported by retrieved context, it outputs exactly:
      "I don't know."

- Search scope control:
  Users can choose where retrieval runs:
      "Library only" | "Uploaded only" | "Both"

- Multi-document disambiguation:
  Users may refer to specific documents by:
      paper1/paper2/...  (library)
      upload1/upload2/... (session uploads)

  Additionally, if the user says "uploaded file" without specifying uploadN:
  - if only one upload exists -> auto-map to upload1
  - if multiple uploads exist -> the app requests an explicit upload selection

- AI fallback:
  If RAG cannot answer from papers (no supporting context),
  the system returns:
      I don't know.
      AI answer (not from the papers): <general answer>

- Verification layers:
  1) Verifier Agent: checks whether the retrieved context contains enough evidence
     to answer the question (FOUND / NOT_FOUND). This reduces hallucinations and
     triggers AI fallback reliably.
  2) Critic Agent (optional): evaluates faithfulness of the final RAG answer with
     respect to the retrieved context and reports a critique.

Performance considerations
--------------------------
LLM calls can exceed API limits if context is too large. To mitigate:
- Context is char-budgeted for QA / Verifier / Critic to avoid 413/TPM errors.
- The app shows a friendly retry message when Groq rate-limits are hit.

Notes on reproducibility
------------------------
- The library index uses HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2").
  This must match the ingestion script.
- The app expects stable metadata fields (paper_id, title, page) to enable correct
  grouping and display.
"""

import os
import re
import json
import base64
import tempfile
from typing import Tuple, List, Optional, Dict, Any

import streamlit as st

# dotenv is optional (avoid crashing if not installed)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Newer LangChain splitters live here (avoids "No module named langchain.text_splitter")
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# CONFIG
# -----------------------------
AGENT_NAME = "ReSearchRAG"
INDEX_DIR = "indexes/faiss_index"
MAX_UPLOADS = 5

# Canonical "unknown" output required by the project spec (strict string match)
IDK_LINE = "I don't know."

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Put it in environment variables or create a .env file with:\n"
        "GROQ_API_KEY=your_real_key_here"
    )

# Guardrails to avoid Groq 413 / TPM bursts (limit prompt sizes deterministically)
QA_MAX_DOCS = 10
VERIFIER_MAX_DOCS = 6
MAX_CONTEXT_CHARS_QA = 12000
MAX_CONTEXT_CHARS_VERIFIER = 6000
PER_DOC_CHAR_CLIP = 1500

# Critic context budget (keep small ,  critic is a secondary call)
CRITIC_MAX_CONTEXT_CHARS = 6000


# -----------------------------
# UTILITIES
# -----------------------------
def _norm(s: str) -> str:
    """Normalize user/model strings for robust comparisons."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _first_reasonable_title(text: str, fallback: str) -> str:
    """
    Best-effort title extraction from raw text.
    Used for uploaded documents (and library catalog display) where the PDF metadata
    may not have a clean title field.
    """
    if not text:
        return fallback
    for line in text.splitlines():
        line = line.strip()
        if 8 <= len(line) <= 180:
            return line
    t = text.strip().replace("\n", " ")
    return (t[:180] + "...") if len(t) > 180 else (t or fallback)


def _strip_paper_refs(q: str) -> str:
    """
    Remove explicit document identifiers (paperN / uploadN) from a query.
    Motivation:
    - Retrieval should focus on content terms, not labels.
    - Document labels are handled separately by the routing/selection logic.
    """
    q2 = re.sub(r"\bpaper\s*\d+\b", " ", q, flags=re.IGNORECASE)
    q2 = re.sub(r"\bupload\s*\d+\b", " ", q2, flags=re.IGNORECASE)
    q2 = re.sub(r"\s+", " ", q2).strip()
    return q2


def _safe_decode_text(file_bytes: bytes) -> str:
    """Decode TXT uploads safely; ignores invalid UTF-8 bytes."""
    try:
        return file_bytes.decode("utf-8")
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore")


def _is_query_too_generic(q: str) -> bool:
    """
    Detect very short / generic queries where retrieval often underperforms.
    The app uses this to inject a slightly more specific fallback query for certain tasks.
    """
    qn = _norm(q)
    return (len(qn) < 18) or (
        qn
        in {
            "final grade",
            "grade",
            "what is it based on",
            "based on",
            "on what is it based on",
        }
    )


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    """
    Remove near-duplicate chunks (by paper+page+prefix) to reduce redundant context.
    """
    seen = set()
    out = []
    for d in docs:
        pid = str(d.metadata.get("paper_id") or d.metadata.get("source") or "")
        page = str(d.metadata.get("page") or "")
        key = (pid, page, d.page_content[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _has_explicit_upload_key(q: str) -> bool:
    """True if the user explicitly referenced uploadN."""
    return bool(re.search(r"\bupload\s*\d+\b", _norm(q)))


def _mentions_uploaded_generic(q: str) -> bool:
    """
    Detect non-specific references like "the uploaded file" that should be resolved
    to upload1 or require disambiguation when multiple uploads exist.
    """
    qn = _norm(q)
    return any(
        p in qn for p in ["uploaded file", "uploaded pdf", "uploaded", "the uploaded", "from the uploaded"]
    )


def _is_rag_idk(text: str) -> bool:
    """
    Robust check for the strict "I don't know" output.
    We keep it strict to avoid false triggers on answers that contain the phrase
    "don't know" as part of normal text.
    """
    t = (text or "").strip()
    tn = _norm(t)
    return tn in {"i don't know.", "i dont know.", "i don't know", "i dont know"}


def _is_groq_limit_error(e: Exception) -> bool:
    """
    Detect Groq request size / rate limit failures.
    When detected, the UI shows a friendly message and asks the user to retry.
    """
    msg = _norm(str(e))
    return any(
        s in msg
        for s in [
            "rate_limit_exceeded",
            "tokens per minute",
            "request too large",
            "413",
            "too large for model",
        ]
    )


def _friendly_limit_message() -> str:
    """User-facing message for Groq rate limit / request size issues."""
    return (
        "⚠️ Groq rate limit / request too large.\n\n"
        "Please click **Ask** again in a few seconds, or reduce the scope (e.g., **Uploaded only** / **Library only**) "
        "or ask a more specific question."
    )


# -----------------------------
# RAG COMPONENT BUILDER
# -----------------------------
@st.cache_resource
def build_rag_components() -> Tuple[Any, Any, Any, Any, Any]:
    """
    Initialize and cache core RAG components.
    Returns:
        base_retriever:
            FAISS retriever over the *library* index.
        qa_chain:
            LLM chain that answers strictly using provided context.
        vectorstore:
            Loaded FAISS vector store (library).
        llm:
            Groq chat model instance.
        verifier_chain:
            A lightweight classifier chain that outputs FOUND / NOT_FOUND.
            Used to decide whether context supports answering before calling QA.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Use higher k so multi-paper questions have more chance to retrieve relevant chunks.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )

    # Main QA prompt (strict IDK line)
    qa_prompt = ChatPromptTemplate.from_template(
        f"You are {AGENT_NAME}, a professional research assistant for academic papers.\n"
        "You must answer ONLY using the provided context.\n"
        f"If the answer is not in the context, say exactly: \"{IDK_LINE}\".\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n"
    )
    qa_chain = qa_prompt | llm | StrOutputParser()

    # Verifier prompt: decides if context contains the answer BEFORE QA runs.
    # Output must be EXACTLY FOUND or NOT_FOUND.
    verifier_prompt = ChatPromptTemplate.from_template(
        "You are a strict verifier.\n"
        "Given CONTEXT and QUESTION, decide if the context contains enough information to answer.\n"
        "Rules:\n"
        "- Output EXACTLY one token: FOUND or NOT_FOUND\n"
        "- If the answer is not explicitly supported by context, output NOT_FOUND\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}\n\n"
        "OUTPUT (FOUND or NOT_FOUND):"
    )
    verifier_chain = verifier_prompt | llm | StrOutputParser()

    return base_retriever, qa_chain, vectorstore, llm, verifier_chain


# -----------------------------
# LIBRARY CATALOG (paper1/paper2 mapping + titles)
# -----------------------------
def _paper_id_sort_key(paper_id: str):
    """
    Sorting helper for stable paper ordering in the sidebar.
    If paper_id already looks like "paperN", keep numeric order;
    otherwise fall back to lexical order.
    """
    m = re.match(r"^paper(\d+)$", str(paper_id).strip().lower())
    if m:
        return (0, int(m.group(1)))
    return (1, str(paper_id).lower())


@st.cache_resource
def build_library_catalog() -> Dict[str, Any]:
    """
    Build a library catalog from the FAISS docstore metadata.
    The catalog is used to:
    - Render an ordered list of library papers in the sidebar
    - Translate between user-facing keys (paper1/paper2/...) and internal paper_id values
    - Enable deterministic compare-mode sampling (doc_ids per paper)

    Returns:
        dict with:
          - ordered: list of paper entries in stable order
          - by_paper_id: map from internal paper_id -> entry
          - paper_key_by_paper_id: map internal id -> "paperX"
          - title_index: list of (normalized_title, paper_key) for title substring matching
    """
    _base_retriever, _qa_chain, vectorstore, _llm, _verifier = build_rag_components()

    paper_groups: Dict[str, List[str]] = {}

    # Group docstore items by paper_id/source so we can form paper-level entries.
    for _, doc_id in vectorstore.index_to_docstore_id.items():
        d: Document = vectorstore.docstore.search(doc_id)
        paper_id = d.metadata.get("paper_id") or d.metadata.get("source") or "unknown"
        paper_groups.setdefault(str(paper_id), []).append(doc_id)

    ordered_paper_ids = sorted(paper_groups.keys(), key=_paper_id_sort_key)

    ordered = []
    by_paper_id: Dict[str, Any] = {}
    paper_key_by_paper_id: Dict[str, str] = {}
    title_index: List[Tuple[str, str]] = []

    for idx, paper_id in enumerate(ordered_paper_ids, start=1):
        doc_ids = paper_groups[paper_id]
        rep_doc: Document = vectorstore.docstore.search(doc_ids[0])

        meta_title = rep_doc.metadata.get("title")
        title = meta_title or _first_reasonable_title(rep_doc.page_content, fallback=str(paper_id))

        pid_norm = str(paper_id).strip().lower()
        if re.match(r"^paper\d+$", pid_norm):
            paper_key = pid_norm
        else:
            paper_key = f"paper{idx}"

        item = {
            "paper_key": paper_key,
            "paper_id": paper_id,
            "title": title,
            "doc_ids": doc_ids,
        }
        ordered.append(item)
        by_paper_id[paper_id] = item
        paper_key_by_paper_id[str(paper_id)] = paper_key
        title_index.append((_norm(title), paper_key))

    return {
        "ordered": ordered,
        "by_paper_id": by_paper_id,
        "paper_key_by_paper_id": paper_key_by_paper_id,
        "title_index": title_index,
    }


def library_key_to_paper_id(paper_key: str) -> Optional[str]:
    """Translate user-facing paper key (paperN) to internal paper_id."""
    cat = build_library_catalog()
    for it in cat["ordered"]:
        if it["paper_key"] == paper_key:
            return it["paper_id"]
    return None


def library_paper_title(paper_key: str) -> Optional[str]:
    """Return the display title for paperN (library)."""
    cat = build_library_catalog()
    for it in cat["ordered"]:
        if it["paper_key"] == paper_key:
            return it["title"]
    return None


def _filter_library_docs_by_key(docs: List[Document], paper_key: str) -> List[Document]:
    """
    Filter a list of retrieved chunks to keep only those belonging to the requested paper.
    """
    paper_id = library_key_to_paper_id(paper_key)
    if not paper_id:
        return []
    out = []
    for d in docs:
        pid = d.metadata.get("paper_id") or d.metadata.get("source")
        if str(pid) == str(paper_id):
            out.append(d)
    return out


# -----------------------------
# CONTEXT BUILDERS (char-budgeted)
# -----------------------------
def build_context_from_docs(
    docs: List[Document],
    max_docs: int,
    max_chars: int,
) -> Tuple[str, Optional[str]]:
    """
    Build the LLM context string from a list of Documents.
    This method is char-budgeted to prevent oversized prompts (Groq 413) and to
    keep token usage predictable.
    Returns:
        context_text:
            Concatenated chunk texts with headers:
                [i] (paper=<id>, page=<page>)
        top_paper_id:
            paper_id of the first included chunk (used for UI labeling)
    """
    if not docs:
        return "", None

    blocks: List[str] = []
    top_paper = None
    total = 0

    for i, d in enumerate(docs[:max_docs]):
        paper_id = d.metadata.get("paper_id") or d.metadata.get("source")
        page = d.metadata.get("page")
        if i == 0:
            top_paper = paper_id

        header = f"[{i+1}] (paper={paper_id}, page={page})\n"
        body = (d.page_content or "").strip()
        body = body[:PER_DOC_CHAR_CLIP]

        block = header + body

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            blocks.append(block[:remaining])
            total += remaining
            break

        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks), top_paper


def build_labeled_context(papers_docs: List[Tuple[str, List[Document]]], per_paper_max: int = 6) -> str:
    """
    Build a labeled multi-document context block, used mainly for compare and summaries.
    Format:
        ===== <label> =====
        [<label> #i | page=p]
        <chunk>
    Labels are human-readable paper titles (Paper A: ..., Paper B: ...).
    """
    out = []
    for label, docs in papers_docs:
        out.append(f"===== {label} =====")
        for i, d in enumerate(docs[:per_paper_max], start=1):
            page = d.metadata.get("page")
            text = (d.page_content or "").strip()[:PER_DOC_CHAR_CLIP]
            out.append(f"[{label} #{i} | page={page}]\n{text}")
        out.append("")
    return "\n\n".join(out).strip()


# -----------------------------
# UPLOADED PAPER INDEXING (multi-upload) - PDF + TXT
# -----------------------------
def _build_uploaded_retriever_pdf(file_name: str, file_bytes: bytes) -> Tuple[Any, str, List[Document]]:
    """
    Create an in-memory FAISS retriever for a single uploaded PDF file.

    Returns:
        retriever: LangChain retriever over in-memory FAISS index
        title: best-effort extracted title for UI display
        chunk_docs: list of chunked Documents (stored for summaries/restore)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    raw_first = pages[0].page_content.strip() if pages else ""
    title = _first_reasonable_title(raw_first, fallback=file_name)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunk_docs = splitter.split_documents(pages)

    # Enrich metadata for consistent downstream filtering and UI labeling.
    for i, d in enumerate(chunk_docs):
        d.metadata["paper_id"] = title
        d.metadata["title"] = title
        d.metadata["source"] = file_name
        d.metadata["page"] = d.metadata.get("page", i)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vstore = FAISS.from_documents(chunk_docs, embeddings)
    retriever = vstore.as_retriever(search_kwargs={"k": 20})
    return retriever, title, chunk_docs


def _build_uploaded_retriever_txt(file_name: str, file_bytes: bytes) -> Tuple[Any, str, List[Document]]:
    """
    Create an in-memory FAISS retriever for a single uploaded TXT file.
    """
    text = _safe_decode_text(file_bytes).strip()
    title = _first_reasonable_title(text[:2000], fallback=file_name)

    base_doc = Document(
        page_content=text,
        metadata={"paper_id": title, "title": title, "source": file_name, "page": 0},
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunk_docs = splitter.split_documents([base_doc])

    for i, d in enumerate(chunk_docs):
        d.metadata["paper_id"] = title
        d.metadata["title"] = title
        d.metadata["source"] = file_name
        d.metadata["page"] = d.metadata.get("page", i)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vstore = FAISS.from_documents(chunk_docs, embeddings)
    retriever = vstore.as_retriever(search_kwargs={"k": 20})
    return retriever, title, chunk_docs


def build_uploaded_retriever(uploaded_file) -> Tuple[Any, str, List[Document]]:
    """
    Dispatch upload indexing depending on file extension (.pdf / .txt).
    Note:
        Unknown extensions are treated as text.
    """
    name = uploaded_file.name
    b = uploaded_file.getvalue()

    if name.lower().endswith(".pdf"):
        return _build_uploaded_retriever_pdf(name, b)
    if name.lower().endswith(".txt"):
        return _build_uploaded_retriever_txt(name, b)

    return _build_uploaded_retriever_txt(name, b)


# -----------------------------
# PAPER RESOLUTION: number or title
# -----------------------------
def resolve_papers_from_query(q: str) -> List[str]:
    """
    Resolve user references to specific documents.
    Returns a list of paper keys:
      - library: "paper1", "paper2", ...
      - uploaded: "upload1", "upload2", ...
    Resolution sources:
    1) Explicit keys in the question text: "paper 2", "upload1", etc.
    2) Title substring matches:
       - library titles from build_library_catalog()
       - uploaded titles from session state
    """
    qn = _norm(q)
    found: List[str] = []

    for m in re.findall(r"\bpaper\s*(\d+)\b", qn):
        found.append(f"paper{m}")
    for m in re.findall(r"\bupload\s*(\d+)\b", qn):
        found.append(f"upload{m}")

    # Title substring matches (library)
    cat = build_library_catalog()
    for title_norm, pkey in cat["title_index"]:
        if title_norm and title_norm in qn:
            found.append(pkey)

    # Title substring matches (uploaded)
    up_map: Dict[str, str] = st.session_state.get("uploaded_number_map", {})
    up_papers: Dict[str, Any] = st.session_state.get("uploaded_papers", {})
    for up_num, upload_id in up_map.items():
        info = up_papers.get(upload_id)
        if not info:
            continue
        tnorm = _norm(info.get("title", ""))
        if tnorm and tnorm in qn:
            found.append(up_num)

    uniq = []
    for x in found:
        if x not in uniq:
            uniq.append(x)
    return uniq


def upload_key_to_upload_id(upload_key: str) -> Optional[str]:
    """Translate user-facing upload key (uploadN) into the session's internal upload_id."""
    return st.session_state.get("uploaded_number_map", {}).get(upload_key)


def upload_title(upload_key: str) -> Optional[str]:
    """Return the display title for uploadN."""
    upload_id = upload_key_to_upload_id(upload_key)
    if not upload_id:
        return None
    info = st.session_state.get("uploaded_papers", {}).get(upload_id)
    return info.get("title") if info else None


# -----------------------------
# INTENT DETECTION
# -----------------------------
def is_greeting(q: str) -> bool:
    """Small heuristic for greeting messages."""
    ql = _norm(q)
    greeting_words = ["hi", "hello", "hey", "שלום", "هلا", "مرحبا"]
    return any(ql == w or ql.startswith(w + " ") for w in greeting_words) and len(ql.split()) <= 6


def is_compare(q: str) -> bool:
    """Detect compare requests."""
    ql = _norm(q)
    return any(w in ql for w in ["compare", "comparison", "difference", "vs", "versus"])


def is_summary_all(q: str) -> bool:
    """Detect 'summarize all papers' requests."""
    ql = _norm(q)
    return any(p in ql for p in ["summary of all", "summarize all", "summary of the papers", "summarize the papers"])


# -----------------------------
# SAVE / LOAD SESSION HELPERS
# -----------------------------
def _export_session_json() -> str:
    """
    Serialize a chat session to JSON, including:
    - chat history
    - search scope
    - critic enable flag
    - uploaded file contents (base64)
    This enables reproducibility and sharing of experiment runs.
    """
    payload: Dict[str, Any] = {
        "version": 1,
        "agent_name": AGENT_NAME,
        "history": st.session_state.get("history", []),
        "query_scope": st.session_state.get("query_scope", "Both"),
        "uploaded_files": [],
        "show_critic": st.session_state.get("show_critic", True),
    }

    up_papers: Dict[str, Any] = st.session_state.get("uploaded_papers", {})
    for upload_id, info in up_papers.items():
        file_name = info.get("file_name") or info.get("source_name") or info.get("source") or ""
        file_type = info.get("file_type") or ("pdf" if file_name.lower().endswith(".pdf") else "txt")
        file_b64 = info.get("file_b64")
        if not file_b64:
            continue
        payload["uploaded_files"].append(
            {
                "upload_id": upload_id,
                "file_name": file_name,
                "file_type": file_type,
                "file_b64": file_b64,
                "title": info.get("title"),
            }
        )

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _reset_chat_state(keep_scope: bool = True):
    """
    Reset conversation state while optionally preserving the search scope setting.
    Used for "New chat" button and when loading sessions.
    """
    scope = st.session_state.get("query_scope", "Both") if keep_scope else "Both"
    show_critic = st.session_state.get("show_critic", True)

    st.session_state.history = []
    st.session_state.question_input = ""
    st.session_state.uploaded_papers = {}
    st.session_state.uploaded_number_map = {}
    st.session_state.query_scope = scope
    st.session_state.show_critic = show_critic


def _load_session_from_json_text(json_text: str) -> Tuple[bool, str]:
    """
    Restore a previously saved session from a JSON string.
    This recreates:
    - chat history
    - search scope
    - critic flag
    - uploaded in-memory indexes (rebuilt from stored base64 file bytes)
    """
    try:
        payload = json.loads(json_text)
    except Exception as e:
        return False, f"Invalid JSON: {e}"

    if not isinstance(payload, dict):
        return False, "Invalid session format."

    _reset_chat_state(keep_scope=False)

    st.session_state.query_scope = payload.get("query_scope", "Both")
    st.session_state.history = payload.get("history", [])
    st.session_state.show_critic = bool(payload.get("show_critic", True))

    uploaded_files = payload.get("uploaded_files", [])
    if isinstance(uploaded_files, list) and uploaded_files:
        rebuilt: Dict[str, Any] = {}
        for item in uploaded_files:
            try:
                upload_id = str(item["upload_id"])
                file_name = str(item["file_name"])
                file_type = str(item.get("file_type", "")).lower().strip()
                file_b64 = str(item["file_b64"])
                file_bytes = base64.b64decode(file_b64.encode("utf-8"))

                if file_type == "pdf" or file_name.lower().endswith(".pdf"):
                    retr, title, docs = _build_uploaded_retriever_pdf(file_name, file_bytes)
                else:
                    retr, title, docs = _build_uploaded_retriever_txt(file_name, file_bytes)

                rebuilt[upload_id] = {
                    "title": title,
                    "retriever": retr,
                    "docs": docs,
                    "file_name": file_name,
                    "file_type": "pdf" if file_name.lower().endswith(".pdf") else "txt",
                    "file_b64": file_b64,
                }
            except Exception:
                continue

        st.session_state.uploaded_papers = rebuilt
        st.session_state.uploaded_number_map = {}
        for i, uid in enumerate(st.session_state.uploaded_papers.keys(), start=1):
            st.session_state.uploaded_number_map[f"upload{i}"] = uid

    return True, "Session loaded."


# -----------------------------
# ANSWER FUNCTION (RAG pipeline)
# -----------------------------
def get_rag_answer_fn():
    """
    Construct and return the main answer function used by the Streamlit UI.
    The returned callable implements:
    - query routing and document resolution (paperN/uploadN/title matching)
    - retrieval over selected scope(s)
    - verifier-based evidence check (FOUND / NOT_FOUND)
    - QA generation from retrieved context
    - AI fallback (general knowledge) when the answer is not found in papers
    - critic evaluation (optional)
    """
    base_retriever, qa_chain, vectorstore, llm, verifier_chain = build_rag_components()

    # Compare-mode prompt
    compare_prompt = ChatPromptTemplate.from_template(
        f"You are {AGENT_NAME}, a professional research assistant.\n"
        "You must answer ONLY using the provided context.\n"
        f"If the context is insufficient to compare, say exactly: \"{IDK_LINE}\".\n\n"
        "TASK: Compare the papers requested by the user.\n"
        "OUTPUT FORMAT:\n"
        "- Paper A (1–2 lines)\n"
        "- Paper B (1–2 lines)\n"
        "- (If Paper C exists) Paper C (1–2 lines)\n"
        "- Similarities (3–6 bullets)\n"
        "- Differences (3–6 bullets)\n"
        "- Practical takeaway (1–2 lines)\n\n"
        "CONTEXT:\n{context}\n\n"
        "USER QUESTION:\n{question}\n"
    )
    compare_chain = compare_prompt | llm | StrOutputParser()

    # Summary prompt (single-paper summarization)
    summary_prompt = ChatPromptTemplate.from_template(
        f"You are {AGENT_NAME}, a professional research assistant.\n"
        "You must answer ONLY using the provided context.\n"
        f"If the context is insufficient, say exactly: \"{IDK_LINE}\".\n\n"
        "TASK: Write a concise summary of this single paper.\n"
        "OUTPUT:\n"
        "- Title: <title>\n"
        "- 3–6 bullet summary\n"
        "- Key contribution (1 line)\n\n"
        "CONTEXT:\n{context}\n\n"
        "USER QUESTION:\n{question}\n"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()

    # AI fallback prompt (no paper context)
    ai_prompt = ChatPromptTemplate.from_template(
        "You are a helpful general assistant.\n"
        "Answer the user question using your general knowledge.\n"
        "Be direct and useful.\n"
        "If you are not sure about something, say so briefly.\n\n"
        "QUESTION:\n{question}\n"
    )
    ai_chain = ai_prompt | llm | StrOutputParser()

    # Critic prompt (faithfulness check)
    critic_prompt = ChatPromptTemplate.from_template(
        "You are the Critic Agent.\n"
        "Your job is to evaluate the faithfulness and correctness of the ANSWER w.r.t. the CONTEXT.\n"
        "Do NOT add new information. Do NOT answer the question again.\n\n"
        "Return a critique in this exact format:\n"
        "Verdict: PASS | FAIL | UNCERTAIN\n"
        "Faithfulness: <1 sentence>\n"
        "Logic: <1 sentence>\n"
        "Evidence: <2-5 bullets, each bullet must cite a context block like [1] or [2] if possible>\n"
        "Issues: <0-5 bullets, only if problems exist>\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER:\n{answer}\n\n"
        "CONTEXT:\n{context}\n"
    )
    critic_chain = critic_prompt | llm | StrOutputParser()

    def _format_ai_fallback(user_q: str) -> str:
        """
        Generate general-knowledge answer after RAG failure.

        Output format is fixed by project requirement:
            I don't know.
            AI answer (not from the papers): ...
        """
        try:
            ai_ans = ai_chain.invoke({"question": user_q}).strip()
        except Exception as e:
            if _is_groq_limit_error(e):
                return _friendly_limit_message()
            return f"Unexpected error: {e}"
        return f"{IDK_LINE}\nAI answer (not from the papers): {ai_ans}"

    def _verdict_found(context: str, question: str) -> bool:
        """Return True if verifier outputs FOUND, else False."""
        if not context.strip():
            return False
        v = (verifier_chain.invoke({"context": context, "question": question}) or "").strip().upper()
        return v == "FOUND"

    def _run_critic(question: str, answer: str, context: str) -> Optional[str]:
        """
        Run critic verification (optional).

        Critic is only meaningful for RAG answers based on paper context.
        It is skipped for AI fallback responses.
        """
        if not st.session_state.get("show_critic", True):
            return None
        if not context.strip():
            return None
        if _is_rag_idk(answer) or answer.strip().startswith(IDK_LINE):
            return None
        try:
            ctx = context[:CRITIC_MAX_CONTEXT_CHARS]
            crit = critic_chain.invoke({"question": question, "answer": answer, "context": ctx}).strip()
            return crit
        except Exception as e:
            if _is_groq_limit_error(e):
                return "Critic: skipped (rate limit / request too large)."
            return f"Critic: error ({e})"

    # Deterministic sampling helpers for compare mode:
    # When the query is generic (e.g., "compare paper1 vs paper2"),
    # semantic retrieval can be empty. Sampling ensures non-empty grounded context.
    def _library_sample_docs(paper_key: str, max_docs: int = 10) -> List[Document]:
        """Fetch representative library chunks directly from the docstore using catalog doc_ids."""
        try:
            cat = build_library_catalog()
            for it in cat["ordered"]:
                if it["paper_key"] == paper_key:
                    doc_ids = it.get("doc_ids", [])[:max_docs]
                    return [vectorstore.docstore.search(did) for did in doc_ids]
        except Exception:
            pass
        return []

    def _upload_sample_docs(upload_key: str, max_docs: int = 10) -> List[Document]:
        """Fetch representative chunks from an uploaded document (stored in session)."""
        upload_id = upload_key_to_upload_id(upload_key)
        info = st.session_state.get("uploaded_papers", {}).get(upload_id) if upload_id else None
        if not info:
            return []
        docs = info.get("docs", []) or []
        return docs[:max_docs]

    def answer_fn(question: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Main answering routine called by the UI.

        Returns:
            answer: str
            source_label: Optional[str]  (paper title/identifier for UI)
            mode: Optional[str]          ("single" | "comparison" | "ai" | None)
            critique: Optional[str]      critic output (if enabled)
        """
        q = question.strip()
        if not q:
            return "", None, None, None

        # Greeting shortcut (no retrieval)
        if is_greeting(q):
            msg = (
                f"Hi! I’m **{AGENT_NAME}**.\n\n"
                "I can help you query and compare your indexed research papers.\n"
                "You can refer to papers by **number** (paper1, paper2, upload1, …) or by **title**.\n\n"
                "Examples:\n"
                "- *What is the main idea of paper2?*\n"
                "- *Summarize all papers*\n"
                "- *Compare paper1 vs upload1*\n"
                "- *Compare “Failure Prediction in Conversational Recommendation Systems” vs paper3*"
            )
            return msg, None, None, None

        # Deterministic title questions (no retrieval)
        m = re.search(r"\btitle\s+of\s+(paper\s*\d+|upload\s*\d+)\b", _norm(q))
        if m:
            key_raw = m.group(1).replace(" ", "")
            key = key_raw

            if key.startswith("paper"):
                title = library_paper_title(key)
                if title:
                    return title, title, "single", None
                return _format_ai_fallback(q), "AI fallback", "ai", None

            if key.startswith("upload"):
                title = upload_title(key)
                if title:
                    return title, title, "single", None
                return _format_ai_fallback(q), "AI fallback", "ai", None

        scope = st.session_state.get("query_scope", "Both")
        query_for_retrieval = _strip_paper_refs(q) or q

        # If user references "uploaded file" generically, resolve or request disambiguation.
        if scope in ("Uploaded only", "Both") and _mentions_uploaded_generic(q) and not _has_explicit_upload_key(q):
            up_map = st.session_state.get("uploaded_number_map", {})
            if up_map:
                if len(up_map) == 1:
                    only_key = list(up_map.keys())[0]
                    q = f"{q} ({only_key})"
                    query_for_retrieval = _strip_paper_refs(q) or q
                else:
                    choices = ", ".join(sorted(up_map.keys()))
                    return (
                        f"You have multiple uploaded files. Please specify which one: {choices}. "
                        f"For example: `What is the final grade based on in upload1?`",
                        None,
                        None,
                        None,
                    )

        # Summarize all papers (library + uploads). Critic is not used here (too noisy).
        if is_summary_all(q):
            parts = []

            cat = build_library_catalog()
            for it in cat["ordered"]:
                pkey = it["paper_key"]
                title = it["title"]

                doc_ids = it["doc_ids"][:10]
                docs = [vectorstore.docstore.search(did) for did in doc_ids]
                ctx = build_labeled_context([(title, docs)], per_paper_max=8)[:MAX_CONTEXT_CHARS_QA]

                try:
                    vctx, _ = build_context_from_docs(
                        docs, max_docs=VERIFIER_MAX_DOCS, max_chars=MAX_CONTEXT_CHARS_VERIFIER
                    )
                    if not _verdict_found(vctx, f"Summarize this paper: {title}"):
                        parts.append(f"### {pkey}: {title}\n{_format_ai_fallback(f'Summarize: {title}')}")
                        continue
                except Exception as e:
                    if _is_groq_limit_error(e):
                        parts.append(f"### {pkey}: {title}\n{_friendly_limit_message()}")
                        continue
                    parts.append(f"### {pkey}: {title}\nUnexpected error: {e}")
                    continue

                try:
                    summ = summary_chain.invoke({"context": ctx, "question": f"Summarize this paper: {title}"}).strip()
                except Exception as e:
                    if _is_groq_limit_error(e):
                        parts.append(f"### {pkey}: {title}\n{_friendly_limit_message()}")
                        continue
                    parts.append(f"### {pkey}: {title}\nUnexpected error: {e}")
                    continue

                if _is_rag_idk(summ):
                    parts.append(f"### {pkey}: {title}\n{_format_ai_fallback(f'Summarize: {title}')}")
                else:
                    parts.append(f"### {pkey}: {title}\n{summ}")

            up_map: Dict[str, str] = st.session_state.get("uploaded_number_map", {})
            up_papers: Dict[str, Any] = st.session_state.get("uploaded_papers", {})
            for up_key, upload_id in up_map.items():
                info = up_papers.get(upload_id)
                if not info:
                    continue
                title = info["title"]
                docs = info.get("docs", [])
                ctx = build_labeled_context([(title, docs)], per_paper_max=8)[:MAX_CONTEXT_CHARS_QA]

                try:
                    vctx, _ = build_context_from_docs(
                        docs, max_docs=VERIFIER_MAX_DOCS, max_chars=MAX_CONTEXT_CHARS_VERIFIER
                    )
                    if not _verdict_found(vctx, f"Summarize this paper: {title}"):
                        parts.append(f"### {up_key}: {title}\n{_format_ai_fallback(f'Summarize: {title}')}")
                        continue
                except Exception as e:
                    if _is_groq_limit_error(e):
                        parts.append(f"### {up_key}: {title}\n{_friendly_limit_message()}")
                        continue
                    parts.append(f"### {up_key}: {title}\nUnexpected error: {e}")
                    continue

                try:
                    summ = summary_chain.invoke({"context": ctx, "question": f"Summarize this paper: {title}"}).strip()
                except Exception as e:
                    if _is_groq_limit_error(e):
                        parts.append(f"### {up_key}: {title}\n{_friendly_limit_message()}")
                        continue
                    parts.append(f"### {up_key}: {title}\nUnexpected error: {e}")
                    continue

                if _is_rag_idk(summ):
                    parts.append(f"### {up_key}: {title}\n{_format_ai_fallback(f'Summarize: {title}')}")
                else:
                    parts.append(f"### {up_key}: {title}\n{summ}")

            if not parts:
                return _format_ai_fallback(q), "AI fallback", "ai", None

            return "\n\n---\n\n".join(parts), None, None, None

        # Compare mode (supports paper1 vs paper2, upload1 vs paper2, etc.)
        if is_compare(q):
            keys = resolve_papers_from_query(q)

            if _mentions_uploaded_generic(q) and not any(k.startswith("upload") for k in keys):
                up_map = st.session_state.get("uploaded_number_map", {})
                if up_map:
                    if len(up_map) == 1:
                        keys.append(sorted(up_map.keys())[0])
                    else:
                        choices = ", ".join(sorted(up_map.keys()))
                        return (
                            f"To compare with an uploaded file, please specify which one: {choices}. "
                            f"For example: `compare upload1 vs paper2`",
                            None,
                            None,
                            None,
                        )

            keys = [k for k in keys if k.startswith("paper") or k.startswith("upload")]
            keys = list(dict.fromkeys(keys))

            if len(keys) < 2:
                return (
                    "To compare, please specify at least two papers (e.g., `paper1 vs paper3` or `upload1 vs paper2`).",
                    None,
                    None,
                    None,
                )

            labeled_docs: List[Tuple[str, List[Document]]] = []

            q_retr = (query_for_retrieval or "").strip()
            q_retr_norm = _norm(q_retr)
            query_is_too_short = (len(q_retr_norm) < 12) or (q_retr_norm in {"compare", "vs", "compare vs"})

            # Build per-paper context pools (up to 3 papers)
            for idx, k in enumerate(keys[:3], start=1):
                label = f"Paper {chr(ord('A') + idx - 1)}"

                # If query is generic, use deterministic sampling to avoid empty retrieval.
                if query_is_too_short:
                    if k.startswith("paper"):
                        title = library_paper_title(k) or k
                        docs_k = _library_sample_docs(k, max_docs=10)
                        labeled_docs.append((f"{label}: {title}", docs_k))
                        continue
                    if k.startswith("upload"):
                        title = upload_title(k) or k
                        docs_k = _upload_sample_docs(k, max_docs=10)
                        labeled_docs.append((f"{label}: {title}", docs_k))
                        continue

                # Otherwise, perform content-based retrieval (and then filter).
                if k.startswith("paper"):
                    title = library_paper_title(k) or k
                    paper_id = library_key_to_paper_id(k)
                    docs_k: List[Document] = []

                    if scope in ("Library only", "Both") and paper_id:
                        pool1 = base_retriever.invoke(f"{title}. {q_retr}") or []
                        pool2 = base_retriever.invoke(title) or []

                        pool = _dedupe_docs(pool1 + pool2)
                        docs_k = [
                            d for d in pool if str(d.metadata.get("paper_id") or d.metadata.get("source")) == str(paper_id)
                        ]

                        if not docs_k:
                            pool3 = base_retriever.invoke(f"{title}. results experiments findings evaluation") or []
                            pool3 = _dedupe_docs(pool3)
                            docs_k = [
                                d for d in pool3 if str(d.metadata.get("paper_id") or d.metadata.get("source")) == str(paper_id)
                            ]

                    # If filtering yields nothing, fall back to deterministic sampling.
                    if not docs_k:
                        docs_k = _library_sample_docs(k, max_docs=10)

                    labeled_docs.append((f"{label}: {title}", docs_k))

                elif k.startswith("upload"):
                    upload_id = upload_key_to_upload_id(k)
                    info = st.session_state.get("uploaded_papers", {}).get(upload_id) if upload_id else None
                    if not info:
                        labeled_docs.append((f"{label}: {k}", []))
                        continue

                    title = info["title"]
                    retr = info["retriever"]

                    docs1 = retr.invoke(q_retr) or []
                    docs2 = retr.invoke(title) or []
                    docs_k = _dedupe_docs(docs1 + docs2)

                    if not docs_k:
                        docs_k = _upload_sample_docs(k, max_docs=10)

                    labeled_docs.append((f"{label}: {title}", docs_k))

            ctx = build_labeled_context(labeled_docs, per_paper_max=6)[:MAX_CONTEXT_CHARS_QA]

            try:
                if not _verdict_found(ctx[:MAX_CONTEXT_CHARS_VERIFIER], q):
                    return _format_ai_fallback(q), "AI fallback", "ai", None
            except Exception as e:
                if _is_groq_limit_error(e):
                    return _friendly_limit_message(), None, None, None
                return f"Unexpected error: {e}", None, None, None

            try:
                ans = compare_chain.invoke({"context": ctx, "question": q}).strip()
            except Exception as e:
                if _is_groq_limit_error(e):
                    return _friendly_limit_message(), None, None, None
                return f"Unexpected error: {e}", None, None, None

            if _is_rag_idk(ans):
                return _format_ai_fallback(q), "AI fallback", "ai", None

            used_titles = []
            for lbl, _docs in labeled_docs:
                used_titles.append(lbl.split(":", 1)[1].strip() if ":" in lbl else lbl)
            ui_label = "comparison: " + " vs ".join(used_titles[:3])

            critique = _run_critic(q, ans, ctx)
            return ans, ui_label, "comparison", critique

        # Normal QA mode (single question answering)
        docs: List[Document] = []
        used_source_label: Optional[str] = None

        requested_keys = resolve_papers_from_query(q)
        requested_keys = [k for k in requested_keys if k.startswith("paper") or k.startswith("upload")]
        requested_keys = list(dict.fromkeys(requested_keys))

        library_pool: List[Document] = []
        if scope in ("Library only", "Both"):
            library_pool = base_retriever.invoke(query_for_retrieval) or []

        uploaded_hits: List[Document] = []
        if scope in ("Uploaded only", "Both"):
            up_papers: Dict[str, Any] = st.session_state.get("uploaded_papers", {})

            # If user explicitly asked uploadN -> retrieve ONLY from those uploads
            if any(k.startswith("upload") for k in requested_keys):
                for k in requested_keys:
                    if not k.startswith("upload"):
                        continue
                    upload_id = upload_key_to_upload_id(k)
                    info = up_papers.get(upload_id) if upload_id else None
                    if not info:
                        continue

                    retr = info["retriever"]
                    title = info.get("title", "")

                    q_base = query_for_retrieval
                    generic = _is_query_too_generic(q_base)

                    hits: List[Document] = []
                    hits += (retr.invoke(q_base) or [])
                    hits += (retr.invoke(f"{title}. {q_base}") or [])

                    if generic:
                        hits += (retr.invoke(f"{title}. grade evaluation based on presentation communication git") or [])
                    else:
                        hits += (retr.invoke("grade evaluation based on") or [])

                    uploaded_hits.extend(_dedupe_docs(hits))

            # Otherwise: search across all uploads
            else:
                for _upload_id, info in up_papers.items():
                    retr = info["retriever"]
                    title = info.get("title", "")
                    q_base = query_for_retrieval
                    generic = _is_query_too_generic(q_base)

                    hits: List[Document] = []
                    hits += (retr.invoke(q_base) or [])
                    hits += (retr.invoke(f"{title}. {q_base}") or [])
                    if generic:
                        hits += (retr.invoke(f"{title}. grade evaluation based on presentation communication git") or [])

                    uploaded_hits.extend(_dedupe_docs(hits))

        if requested_keys:
            filtered: List[Document] = []
            for k in requested_keys:
                if k.startswith("paper"):
                    filtered.extend(_filter_library_docs_by_key(library_pool, k))
                elif k.startswith("upload"):
                    filtered.extend(uploaded_hits)
            docs = filtered

            # Second attempt: use title as hint if nothing found (library)
            if not docs and scope in ("Library only", "Both"):
                for k in requested_keys:
                    if not k.startswith("paper"):
                        continue
                    title = library_paper_title(k)
                    if title:
                        pool2 = base_retriever.invoke(f"{title}. {query_for_retrieval}") or []
                        docs.extend(_filter_library_docs_by_key(pool2, k))
        else:
            if scope in ("Library only", "Both"):
                docs.extend(library_pool)
            if scope in ("Uploaded only", "Both"):
                docs.extend(uploaded_hits)

        if not docs:
            return _format_ai_fallback(q), "AI fallback", "ai", None

        qa_ctx, top_paper_id = build_context_from_docs(docs, max_docs=QA_MAX_DOCS, max_chars=MAX_CONTEXT_CHARS_QA)
        verifier_ctx, _ = build_context_from_docs(docs, max_docs=VERIFIER_MAX_DOCS, max_chars=MAX_CONTEXT_CHARS_VERIFIER)

        try:
            if not _verdict_found(verifier_ctx, q):
                return _format_ai_fallback(q), "AI fallback", "ai", None
        except Exception as e:
            if _is_groq_limit_error(e):
                return _friendly_limit_message(), None, None, None
            return f"Unexpected error: {e}", None, None, None

        try:
            ans = qa_chain.invoke({"question": q, "context": qa_ctx}).strip()
        except Exception as e:
            if _is_groq_limit_error(e):
                return _friendly_limit_message(), None, None, None
            return f"Unexpected error: {e}", None, None, None

        if _is_rag_idk(ans):
            return _format_ai_fallback(q), "AI fallback", "ai", None

        # UI label: attempt to map top chunk paper_id -> a paper key -> title
        if top_paper_id:
            cat = build_library_catalog()
            pkey = cat["paper_key_by_paper_id"].get(str(top_paper_id))
            if pkey:
                used_source_label = library_paper_title(pkey) or str(top_paper_id)
            else:
                used_source_label = str(top_paper_id)

        critique = _run_critic(q, ans, qa_ctx)
        return ans, used_source_label, "single", critique

    return answer_fn


# -----------------------------
# UI (Streamlit)
# -----------------------------
def main():
    """
    Streamlit entry point.

    The UI is split into:
    - Sidebar: controls, upload/index actions, scope selection, paper catalog
    - Main area: conversation history + input box

    Session state is used to keep:
    - chat history
    - uploaded document indexes and numbering map
    - user preferences (scope, critic enabled)
    """
    st.set_page_config(page_title=AGENT_NAME, layout="wide")

    # Session state init (persistent across reruns in Streamlit)
    if "history" not in st.session_state:
        st.session_state.history = []
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "query_scope" not in st.session_state:
        st.session_state.query_scope = "Both"
    if "show_critic" not in st.session_state:
        st.session_state.show_critic = True

    # Multi-upload store: maps internal upload_id -> {title, retriever, docs, ...}
    if "uploaded_papers" not in st.session_state:
        st.session_state.uploaded_papers = {}
    # Maps user-facing keys (upload1/upload2/...) -> internal upload_id
    if "uploaded_number_map" not in st.session_state:
        st.session_state.uploaded_number_map = {}

    rag_answer = get_rag_answer_fn()

    # Sidebar
    with st.sidebar:
        st.header(f"{AGENT_NAME} Settings")

        if st.button("New chat"):
            _reset_chat_state(keep_scope=True)
            st.rerun()

        st.subheader("Save / Load chat")

        session_json = _export_session_json()
        st.download_button(
            label="Save chat/session (JSON)",
            data=session_json.encode("utf-8"),
            file_name="researchrag_session.json",
            mime="application/json",
        )

        loaded_json_file = st.file_uploader(
            "Load saved session (JSON)",
            type=["json"],
            accept_multiple_files=False,
            help="Upload a previously saved session JSON to restore chat + uploaded files.",
            key="session_json_uploader",
        )
        if loaded_json_file is not None:
            if st.button("Load session now"):
                ok, msg = _load_session_from_json_text(
                    loaded_json_file.getvalue().decode("utf-8", errors="ignore")
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("---")

        st.markdown(
            "- Query your indexed research library\n"
            "- Upload PDF/TXT and chat with them instantly\n"
            "- Control where the search runs"
        )

        st.subheader("Critic Agent")
        st.session_state.show_critic = st.checkbox(
            "Enable Critic verification",
            value=bool(st.session_state.get("show_critic", True)),
            help="Adds a Critic layer to evaluate whether the answer is faithful to the retrieved context.",
        )

        st.markdown("---")

        uploaded_files = st.file_uploader(
            "Upload PDF/TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help=f"Up to {MAX_UPLOADS} files per session.",
            key="paper_uploader",
        )

        if uploaded_files:
            if len(uploaded_files) > MAX_UPLOADS:
                st.warning(f"Please upload at most {MAX_UPLOADS} files.")
            else:
                if st.button("Index uploaded file(s)"):
                    with st.spinner("Indexing uploaded files..."):
                        for uf in uploaded_files:
                            # Internal ID prevents collisions and helps persistence in session JSON.
                            upload_id = f"uploaded::{uf.name}"

                            # Skip if already indexed
                            if upload_id in st.session_state.uploaded_papers:
                                continue

                            retr, title, docs = build_uploaded_retriever(uf)

                            # Persist file content in base64 to support "Save session"
                            b = uf.getvalue()
                            b64 = base64.b64encode(b).decode("utf-8")
                            ftype = "pdf" if uf.name.lower().endswith(".pdf") else "txt"

                            st.session_state.uploaded_papers[upload_id] = {
                                "title": title,
                                "retriever": retr,
                                "docs": docs,
                                "file_name": uf.name,
                                "file_type": ftype,
                                "file_b64": b64,
                            }

                        # Rebuild user-facing upload numbering map
                        st.session_state.uploaded_number_map = {}
                        for i, uid in enumerate(st.session_state.uploaded_papers.keys(), start=1):
                            st.session_state.uploaded_number_map[f"upload{i}"] = uid

                    st.success(f"Indexed {len(uploaded_files)} uploaded file(s).")

        st.subheader("Search scope")
        st.session_state.query_scope = st.radio(
            "Where should it search?",
            ["Library only", "Uploaded only", "Both"],
            index=2,
        )

        st.subheader("Library papers")
        cat = build_library_catalog()
        for it in cat["ordered"]:
            st.write(f"- **{it['paper_key']}**: {it['title']}")

        if st.session_state.uploaded_number_map:
            st.subheader("Uploaded papers")
            for up_key, upload_id in st.session_state.uploaded_number_map.items():
                info = st.session_state.uploaded_papers.get(upload_id)
                if info:
                    st.write(f"- **{up_key}**: {info['title']}")

    # Main UI
    st.title(f"📚 {AGENT_NAME}")
    st.write(
        "Query your indexed research library and uploaded PDFs/TXT. "
        "The system retrieves relevant passages and answers strictly from those sources."
    )

    st.subheader("Conversation")

    if st.session_state.history:
        for turn in st.session_state.history:
            with st.container():
                st.markdown(f"**You:** {turn['question']}")

                label = turn.get("source_label")
                mode = turn.get("mode")

                if mode == "comparison" and label:
                    st.markdown(f"**{AGENT_NAME}** *({label})*:")
                elif mode == "single" and label:
                    st.markdown(f"**{AGENT_NAME}** *(using {label})*:")
                elif mode == "ai":
                    st.markdown(f"**{AGENT_NAME}** *(AI fallback — not from papers)*:")
                else:
                    st.markdown(f"**{AGENT_NAME}:**")

                st.write(turn["answer"])

                crit = turn.get("critique")
                if crit:
                    with st.expander("Critic verification", expanded=False):
                        st.write(crit)

                st.markdown("---")
    else:
        st.info("Ask your first question to start the conversation.")

    st.subheader("Ask a new question")
    st.text_area(
        "Your question",
        key="question_input",
        placeholder="e.g., Compare paper1 vs upload1, or Summarize all papers, or What is the main idea of paper2?",
        height=90,
    )

    def on_ask():
        """Handle Ask button click: run RAG pipeline and append turn to chat history."""
        q = st.session_state.question_input.strip()
        if not q:
            return
        with st.spinner("Thinking..."):
            answer, source_label, mode, critique = rag_answer(q)

        st.session_state.history.append(
            {"question": q, "answer": answer, "source_label": source_label, "mode": mode, "critique": critique}
        )
        st.session_state.question_input = ""

    st.button("Ask", type="primary", on_click=on_ask)


if __name__ == "__main__":
    main()

#  .venv\Scripts\activate
#  streamlit run app.py