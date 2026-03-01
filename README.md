# ReSearchRAG: Academic Paper Research Assistant (RAG)

ReSearchRAG is an interactive Retrieval-Augmented Generation (RAG) application designed to answer questions **grounded strictly** in a collection of academic papers. It supports:

- A **persistent library** of papers (pre-indexed FAISS index on disk)
- **Session uploads** (PDF/TXT indexed in-memory per session)
- A **search scope router** (Library only / Uploaded only / Both)
- A **Verifier** + optional **Critic** layer to improve faithfulness and transparency
- A safe **AI fallback** when the answer is not found in the documents

---

## Core Features

- **Strict Grounding & Hallucination Mitigation**
  - The assistant answers **only** using retrieved context.
  - If the answer is not supported by context, it outputs exactly:
    - `I don't know.`
    - then provides: `AI answer (not from the papers): ...`
  - A **Verifier Agent** checks whether the retrieved context is sufficient before generating an answer.

- **Dynamic Search Scope**
  - Route queries to: **Library only**, **Uploaded only**, or **Both**.

- **Multi-Document Disambiguation**
  - Users can explicitly target documents:
    - Library: `paper1`, `paper2`, ...
    - Uploads: `upload1`, `upload2`, ...

- **Compare Mode**
  - Structured comparisons such as: `Compare paper1 vs upload1`.

- **Critic Agent (Optional)**
  - Provides a critique assessing whether the answer is faithful to the retrieved context.

- **CLI + GUI**
  - Streamlit UI (`app.py`) for full experience
  - CLI tool (`query.py`) for quick testing without a browser

---

## Project Structure

- `app.py`  
  Main Streamlit web application (UI + routing + RAG pipeline + Verifier + Critic + AI fallback).

- `ingest.py`  
  Parses PDFs in `data/papers/`, chunks them, embeds them, and builds the **persistent** FAISS index at `indexes/faiss_index/`.

- `query.py`  
  CLI utility for testing the pre-built library index quickly.

- `data/papers/`  
  Folder containing the **library PDF papers** (persistent corpus).

- `indexes/faiss_index/`  
  The saved FAISS vectorstore created by `ingest.py`.

- `tests/`  
  Small tests used during development (comparison behavior, limits, etc.).

- `.env`  
  Stores `GROQ_API_KEY=...`

- `requirements.txt`  
  Python dependencies.

---

## Setup Instructions (Recommended: VS Code)

### 1) Prerequisites
- **Python 3.8+**
- A **Groq API Key** (create one at console.groq.com)

### 2) Open the Project
Open the project folder in **VS Code** and open a terminal:
`Terminal → New Terminal`

> Run the commands below from the **project root** (the folder that contains `app.py`).

### 3) Create & Activate a Virtual Environment

**Windows**

python -m venv .venv

.\.venv\Scripts\activate

**macOS/Linux**

python3 -m venv .venv

source .venv/bin/activate

### 4) Install Dependencies

pip install -r requirements.txt


### 5) Configure Environment Variables
Create a file called .env in the project root and add:

GROQ_API_KEY=your_actual_api_key_here

--------------------------------------------------------------------------------------------------------------------

## Populating the Library (Adding Papers)

 **Step 1:**  Add PDFs to the Library Folder 

    Put your academic PDFs into:
      data/papers/

    Example: data/papers/paper1.pdf
      data/papers/paper2.pdf

 **Step 2:** Build / Rebuild the FAISS Index (Required!)

    Run the ingestion script:

      python ingest.py

    You should see console output indicating that PDFs were loaded, chunked, embedded, and saved into: 
    indexes/faiss_index/

**If you add/remove/rename any PDF inside data/papers/, you must re-run: python ingest.py , Otherwise, the app will still use the old index.**

**This repository does not include any PDF papers or FAISS indexes. Add your own PDFs to data/papers/ and run python ingest.py**

--------------------------------------------------------------------------------------------------------------------

## Running the System

###  Launch the Streamlit Web App (Main UI)
    streamlit run app.py

Streamlit typically opens the app at: http://localhost:8501

--------------------------------------------------------------------------------------------------------------------


## Using the App (Main Workflow)

### 1) Choose Search Scope

In the sidebar:

Library only → searches the persistent index
Uploaded only → searches only session uploads
Both → searches both and routes intelligently

### 2) Upload Files (Session Documents)

Use the sidebar uploader to upload PDF/TXT files
Click Index uploaded file(s) to build the in-memory FAISS index
The uploads will be labeled in the sidebar as:
upload1, upload2, ...

### 3) Ask Questions

Examples:

What is the main contribution of paper2?
Summarize upload1
Compare paper1 vs upload1
From upload2: what is the evaluation protocol?

--------------------------------------------------------------------------------------------------------------------

## Important Note About Prompting (fun but true 😄)

This is a RAG system — so tiny changes in your wording can lead to very different retrieval results.

To get the best answers:

Be specific

Mention the paper tag (paper2, upload1) or part of the title

If the answer isn’t good on the first try, rephrase and try again

Think of it like talking to a very smart librarian:
if you say “give me the thing about the thing”… the librarian will stare at you 🤨
but if you say “paper3, experiments section, what metric improved most?” — boom 💥

--------------------------------------------------------------------------------------------------------------------

## Evaluation & Testing Guide (Suggested)

### 1) Out-of-Scope Question (Tests strict fallback)

Query:
What is the capital of France?
Expected behavior:
The system outputs:
I don't know.
then: AI answer (not from the papers): ...

### 2) Uploading & Routing
Set scope to Uploaded only
Upload a PDF and click Index uploaded file(s)
Ask a question specific to that upload

### 3) Cross-Document Comparison

Query:
Compare paper1 vs upload1
Expected behavior:
Structured comparison (similarities, differences, takeaway)

--------------------------------------------------------------------------------------------------------------------

## Troubleshooting

**“GROQ_API_KEY is not set”**

Make sure .env exists in the project root and contains:
GROQ_API_KEY=...

**“Index not found at indexes/faiss_index”**

You must build the library index first:
python ingest.py

**“Groq rate limit / request too large”**

Wait a few seconds and click Ask again
Or reduce scope (Library only / Uploaded only)
Or ask a more specific question

--------------------------------------------------------------------------------------------------------------------

## Dependencies / Models (Implementation Notes)

Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector Store: FAISS
LLM: Groq llama-3.1-8b-instant
PDF Loader: PyPDFLoader


--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------

## Authors: Yousef Shihade , Fadi Kaiss 
## University of Haifa - Department of Computer Science

--------------------------------------------------------------------------------------------------------------------

### This project was developed as a university RAG system project for academic evaluation and demonstration.