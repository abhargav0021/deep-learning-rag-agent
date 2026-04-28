"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Layout:
  - Left sidebar: Corpus ingestion and document library
  - Centre: Document chunk viewer
  - Right: Conversational query interface

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager

# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Page title */
        .page-title {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #f0f0f0;
            margin: 0;
            padding: 0;
        }

        .page-subtitle {
            font-size: 0.82rem;
            color: #888;
            margin-top: 0.2rem;
            letter-spacing: 0.01em;
        }

        .page-header {
            border-bottom: 1px solid #2a2a2a;
            padding-bottom: 1rem;
            margin-bottom: 1.5rem;
        }

        /* Section labels */
        .section-label {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 0.6rem;
        }

        /* Corpus document card */
        .doc-card {
            background: #111118;
            border: 1px solid #222;
            border-left: 3px solid var(--card-accent, #3B82F6);
            border-radius: 4px;
            padding: 9px 12px;
            margin-bottom: 6px;
        }

        .doc-card-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #e0e0e0;
            font-family: 'JetBrains Mono', monospace;
        }

        .doc-card-meta {
            font-size: 0.72rem;
            color: #666;
            margin-top: 2px;
        }

        .topic-badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            border-radius: 3px;
            padding: 1px 6px;
            margin-right: 6px;
        }

        /* Chat */
        .chat-welcome {
            background: #111118;
            border: 1px solid #1e1e2e;
            border-radius: 6px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        .chat-welcome-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #bbb;
            margin-bottom: 0.5rem;
        }

        .source-list {
            font-size: 0.8rem;
            color: #888;
            font-family: 'JetBrains Mono', monospace;
        }

        .stat-pill {
            display: inline-block;
            background: #1a1a2a;
            border: 1px solid #2a2a3a;
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 0.72rem;
            color: #aaa;
            margin-right: 6px;
            margin-bottom: 4px;
        }

        /* Chunk viewer */
        .chunk-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #bbb;
            background: #0d0d14;
            border: 1px solid #1e1e2e;
            border-radius: 4px;
            padding: 10px 12px;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        .chunk-label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #555;
            margin-bottom: 4px;
        }

        /* Divider */
        hr {
            border-color: #1e1e2e !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar — Ingestion
# ---------------------------------------------------------------------------

TOPIC_COLORS: dict[str, str] = {
    "ANN": "#4A90D9",
    "CNN": "#E67E22",
    "RNN": "#27AE60",
    "LSTM": "#8E44AD",
    "Seq2Seq": "#E74C3C",
    "Autoencoder": "#16A085",
    "GAN": "#F39C12",
    "SOM": "#1ABC9C",
    "general": "#555",
}


def render_ingestion_panel(store: VectorStoreManager, chunker: DocumentChunker) -> None:
    st.sidebar.markdown('<div class="section-label">Corpus Ingestion</div>', unsafe_allow_html=True)

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.sidebar.button("Ingest Documents", use_container_width=True):
            import tempfile

            file_paths = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.getvalue())
                file_paths.append((Path(tmp.name), file.name))

            progress = st.sidebar.progress(0, text="Chunking files...")
            chunks = chunker.chunk_files(file_paths)
            progress.progress(50, text="Embedding and storing...")
            result = store.ingest(chunks)
            progress.progress(100, text="Complete.")
            progress.empty()

            st.sidebar.success(
                f"{result.ingested} chunks added — {result.skipped} duplicates skipped."
            )
            if result.errors:
                st.sidebar.error(f"Errors: {result.errors}")


def render_corpus_stats(store: VectorStoreManager) -> None:
    try:
        count = store._collection.count()
        if count == 0:
            return
        all_meta = store._collection.get(include=["metadatas"])
        topics = sorted({m.get("topic", "?") for m in all_meta["metadatas"]})

        st.sidebar.divider()
        st.sidebar.markdown('<div class="section-label">Corpus Overview</div>', unsafe_allow_html=True)

        pills_html = "".join(
            f'<span class="stat-pill">{t}</span>' for t in topics
        )
        st.sidebar.markdown(
            f'<div style="margin-bottom:6px">'
            f'<span class="stat-pill">{count} chunks</span>'
            f'<span class="stat-pill">{len(topics)} topics</span>'
            f"</div>"
            f"<div>{pills_html}</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def render_ingested_documents_panel(store: VectorStoreManager) -> None:
    try:
        docs = store.list_documents()
        if not docs:
            return

        st.sidebar.divider()
        with st.sidebar.expander(f"Document Library  ({len(docs)})", expanded=False):
            for doc in docs:
                source = doc["source"]
                topic = doc["topic"]
                chunk_count = doc["chunk_count"]
                color = TOPIC_COLORS.get(topic, "#555")

                st.markdown(
                    f"""
                    <div class="doc-card" style="--card-accent: {color}">
                        <div class="doc-card-title">{source}</div>
                        <div class="doc-card-meta">
                            <span class="topic-badge" style="background:{color}18; color:{color};">{topic}</span>
                            {chunk_count} chunks
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_view, col_del = st.columns([1, 1])
                with col_view:
                    if st.button("View", key=f"view_{source}", use_container_width=True):
                        st.session_state["selected_document"] = source
                with col_del:
                    if st.button("Remove", key=f"del_{source}", use_container_width=True):
                        store.delete_document(source)
                        st.rerun()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Centre — Document Viewer
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    st.markdown('<div class="section-label">Document Viewer</div>', unsafe_allow_html=True)

    try:
        docs = store._collection.get(include=["metadatas", "documents"])
    except Exception:
        st.info("No documents found.")
        return

    if not docs or not docs.get("documents"):
        st.markdown(
            '<p style="color:#555; font-size:0.85rem;">No documents ingested yet. '
            "Upload study materials from the sidebar to begin.</p>",
            unsafe_allow_html=True,
        )
        return

    documents = docs["documents"]
    metadatas = docs["metadatas"]

    grouped: dict = defaultdict(list)
    for text, meta in zip(documents, metadatas):
        source = meta.get("source", "Unknown")
        grouped[source].append((text, meta))

    total_chunks = len(documents)
    total_docs = len(grouped)

    st.markdown(
        f'<p class="doc-card-meta" style="margin-bottom:0.75rem;">'
        f"{total_docs} documents &middot; {total_chunks} chunks</p>",
        unsafe_allow_html=True,
    )

    for source, chunk_list in grouped.items():
        topic = chunk_list[0][1].get("topic", "?")
        difficulty = chunk_list[0][1].get("difficulty", "?")
        is_selected = st.session_state.get("selected_document") == source
        label = f"{source}  [{topic} | {difficulty}]  {len(chunk_list)} chunks"

        with st.expander(label, expanded=is_selected):
            for i, (text, meta) in enumerate(chunk_list):
                st.markdown(
                    f'<div class="chunk-label">Chunk {i + 1}</div>'
                    f'<div class="chunk-text">{text[:400]}{"..." if len(text) > 400 else ""}</div>',
                    unsafe_allow_html=True,
                )
                if i < len(chunk_list) - 1:
                    st.markdown("<hr>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Right — Chat Interface
# ---------------------------------------------------------------------------


def _is_insufficient(text: str) -> bool:
    return any(
        phrase in text.lower()
        for phrase in [
            "does not contain enough information",
            "not enough information",
            "cannot answer",
            "no relevant",
            "does not mention",
        ]
    )


def render_chat_panel(graph) -> None:
    from langchain_core.messages import HumanMessage

    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown('<div class="section-label">Interview Preparation</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.8rem; color:#666; margin-top:-4px;">'
            "Answers are grounded in the ingested corpus. Topics: ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder.</p>",
            unsafe_allow_html=True,
        )
    with col_btn:
        st.write("")
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if not st.session_state.chat_history:
        example_questions = [
            "What is backpropagation?",
            "How do CNNs detect features in images?",
            "What is the vanishing gradient problem?",
            "How does an LSTM retain long-term dependencies?",
        ]

        st.markdown(
            '<div class="chat-welcome">'
            '<div class="chat-welcome-title">Suggested questions</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        for i, q in enumerate(example_questions):
            col = col1 if i % 2 == 0 else col2
            if col.button(q, use_container_width=True, key=f"example_{i}"):
                st.session_state["_pending_query"] = q
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("timestamp"):
                st.caption(msg["timestamp"])
            if msg["role"] == "assistant":
                insufficient = _is_insufficient(msg["content"])
                if msg.get("sources") and not msg.get("no_context_found") and not insufficient:
                    with st.expander("Sources"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<span class="source-list">{src}</span>',
                                unsafe_allow_html=True,
                            )

    query = st.chat_input("Enter your question...")

    if "_pending_query" in st.session_state:
        query = st.session_state.pop("_pending_query")

    if query:
        st.chat_message("user").write(query)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        with st.spinner("Generating response..."):
            result = graph.invoke(
                {"messages": [HumanMessage(content=query)]},
                config=config,
            )

        final = result.get("final_response")
        if final:
            answer = final.answer
            sources = list(dict.fromkeys(final.sources))
            no_context = final.no_context_found
            confidence = final.confidence
            rewritten_query = final.rewritten_query
        else:
            answer = "No response generated."
            sources = []
            no_context = False
            confidence = None
            rewritten_query = None

        insufficient = _is_insufficient(answer)
        with st.chat_message("assistant"):
            st.write(answer)
            if sources and not no_context and not insufficient:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(
                            f'<span class="source-list">{src}</span>',
                            unsafe_allow_html=True,
                        )
            if no_context or insufficient:
                st.info("No relevant content found in the corpus for this query.")

        ts = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append({"role": "user", "content": query, "timestamp": ts})
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "no_context_found": no_context,
                "confidence": confidence,
                "rewritten_query": rewritten_query,
                "original_query": query,
                "timestamp": ts,
            }
        )
        st.rerun()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()

    st.markdown(
        '<div class="page-header">'
        f'<div class="page-title">{settings.app_title}</div>'
        '<div class="page-subtitle">'
        "RAG-powered interview preparation &mdash; LangChain &middot; LangGraph &middot; ChromaDB"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    if store._collection.count() == 0:
        corpus_dir = Path(settings.corpus_dir)
        md_files = list(corpus_dir.glob("*.md"))
        if md_files:
            with st.spinner("Loading corpus..."):
                file_pairs = [(p, p.name) for p in md_files]
                chunks = chunker.chunk_files(file_pairs)
                store.ingest(chunks)

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)
    render_ingested_documents_panel(store)

    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_panel(graph)


if __name__ == "__main__":
    main()
