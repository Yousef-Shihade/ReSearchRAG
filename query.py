"""
query.py

This script provides a *command-line interface (CLI)* for querying the pre-built
FAISS paper library index (built by ingest/build_library_index.py).

Why this file exists
--------------------
During development, a CLI is extremely useful for:
- Quickly testing retrieval quality without running Streamlit
- Debugging whether the FAISS index loads correctly
- Inspecting which papers/pages are being retrieved
- Iterating on retriever settings (k, MMR, fetch_k, etc.)

Core approach
-------------
1) Load the on-disk FAISS index with the same embedding model used at indexing time.
2) Retrieve relevant chunks using an MMR retriever (diversity-aware).
3) Optionally auto-select a dominant paper from the retrieved chunks.
4) Ask an LLM (Groq) to answer *strictly from context*, with an explicit "I don't know" fallback.

Important note
--------------
This CLI script uses an *interactive* disambiguation step (input()) when multiple papers
appear relevant. The Streamlit application implements a different, UI-friendly approach,
but the idea is similar: do not silently answer from the wrong paper when ambiguity exists.
"""

import os
from collections import Counter
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

INDEX_DIR = "indexes/faiss_index"


def load_vectorstore():
    """
    Load the FAISS vector store from disk.

    Requirements:
    - The FAISS index must already exist at INDEX_DIR.
    - The embedding model must match the one used during indexing,
      otherwise vector comparisons become meaningless.
    """

    # Same embedding model as in ingest.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.isdir(INDEX_DIR):
        raise RuntimeError(f"Index not found at {INDEX_DIR}. Run ingest.py first.")

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def build_rag_answer_fn(vectorstore):
    """
    Build and return a function that answers questions using:
    - retrieval from the FAISS index
    - an LLM constrained to retrieved context (RAG)

    Returns:
        answer_fn(question: str) -> str
    """

    # Retrieval policy: MMR (Max Marginal Relevance)
    # MMR improves diversity by avoiding near-duplicate chunks in the final context,
    # which often increases coverage for multi-facet questions.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,          # chunks fed into the LLM
            "fetch_k": 25,   # pool of candidate chunks
            "lambda_mult": 0.5,  # relevance-diversity tradeoff (higher -> more relevance, lower -> more diversity)
        },
    )

    
    # Prompt template:
    #  - Forces answers to be grounded in retrieved context
    #  - Adds traceability by requiring paper/page references in the response
    prompt_tmpl = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful research assistant for reading research papers. "
            "Answer ONLY using the provided context. "
            "If the answer is not in the context, say exactly: 'I don't know'. "
            "When you answer, explicitly mention which paper and page(s) you used, "
            "for example: 'According to paper=rag_survey, page=12, ...'."
        ),
        (
            "human",
            "Paper focus: {paper_name}\n"
            "Question: {question}\n\n"
            "Context:\n{context}"
        ),
    ])

    # LLM backend:
    # Groq-hosted model with slightly low temperature to reduce hallucinations.
    # (Temperature is not a guarantee of factuality; the context constraint is the real safety mechanism.)
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )

    parser = StrOutputParser()

    # Chain: (question, context, paper_name) -> prompt -> LLM -> string
    chain = prompt_tmpl | llm | parser

    def format_context(docs):
        """
        Convert retrieved Document chunks into a formatted context block.

        Including paper/page metadata provides traceability for both:
        - the user (they can validate claims)
        - the developer (debugging incorrect citations)
        """
        parts = []
        for i, d in enumerate(docs, start=1):
            paper = d.metadata.get("paper_name", "unknown_paper")
            page = d.metadata.get("page", "?")
            parts.append(
                f"[{i}] (paper={paper}, page={page})\n{d.page_content}"
            )
        return "\n\n".join(parts)


    def select_paper(docs) -> str | None:
        """
        Decide whether to focus on a single paper or keep multiple.

        Rationale:
        - When retrieved chunks clearly come from one paper, focusing improves coherence
          and reduces the chance of mixing unrelated evidence.
        - When retrieval is ambiguous across papers, the CLI asks the user to clarify.

        Returns:
            - paper_name (str) if a single paper should be used
            - None if the answer should use chunks from multiple papers
        """
        paper_names = [
            d.metadata.get("paper_name")
            for d in docs
            if d.metadata.get("paper_name") is not None
        ]

        if not paper_names:
            return None

        counts = Counter(paper_names)
        # If only one paper appears, we can auto-select it
        if len(counts) == 1:
            return paper_names[0]

        # Get top 2
        most_common = counts.most_common()
        (best_paper, best_count) = most_common[0]
        second_count = most_common[1][1] if len(most_common) > 1 else 0

        #  Auto selection logic (Option 1)
        if best_count >= 2 and best_count >= second_count * 1.5:
            # Clearly one paper dominates
            return best_paper

        #  Ambiguous → ask user (Option 3)
        print("\nI found multiple papers that might match your question:")
        unique_papers = list(counts.keys())
        for idx, name in enumerate(unique_papers, start=1):
            print(f"  {idx}. {name} (score ~{counts[name]})")

        while True:
            choice = input(
                "Which paper do you want me to focus on? "
                "(type number or part of the name, or press Enter to use all): "
            ).strip()

            # Empty input → use all papers
            if choice == "":
                return None

            # If they typed a number
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(unique_papers):
                    return unique_papers[idx - 1]
                else:
                    print("Invalid number, try again.")
                    continue

            # Try match by substring
            lowered = choice.lower()
            matches = [n for n in unique_papers if lowered in n.lower()]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                print("That matches multiple papers, please be more specific.")
            else:
                print("Didn't recognize that paper name, try again.")

    def answer_fn(question: str) -> str:
        """
        Answer a user question using retrieval + LLM.

        Steps:
        1) Retrieve candidate chunks from the FAISS index
        2) Decide whether to focus on a single paper (optional)
        3) Build a context string including citations (paper/page)
        4) Generate answer constrained to that context
        """
        # 1) Retrieve candidate chunks
        docs = retriever.invoke(question)
        if not docs:
            # Note: This message is more verbose than the strict "I don't know"
            # requirement used in the Streamlit app. For CLI debugging, we keep
            # a short explanation.
            return "I don't know. I couldn't find any relevant passages in the indexed papers."

        # 2) Choose paper focus (auto-select or ask user if ambiguous)
        selected_paper = select_paper(docs)

        # 3) Filter docs if a single paper was selected
        if selected_paper is not None:
            used_docs = [d for d in docs if d.metadata.get("paper_name") == selected_paper]
            if not used_docs:
            # If filtering removed everything (edge case), fall back to original docs
                used_docs = docs
        else:
            used_docs = docs

        # 4) Build context and call the LLM
        context = format_context(used_docs)
        result = chain.invoke({
            "question": question,
            "context": context,
            "paper_name": selected_paper or "multiple_papers",
        })
        return result

    return answer_fn


def main():
    """
    CLI main loop.

    Usage:
    - Ensure GROQ_API_KEY is set (e.g., via .env).
    - Ensure FAISS index exists (run ingest/build_library_index.py first).
    - Run:
        python query.py
    - Ask questions interactively.
    """
    load_dotenv()  # Load GROQ_API_KEY

    vectorstore = load_vectorstore()
    rag_answer = build_rag_answer_fn(vectorstore)

    print("🔍 Research RAG Agent – ask questions about your papers.")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Your question: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        ans = rag_answer(q)
        print("\nAnswer:\n")
        print(ans)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()

