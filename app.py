import streamlit as st

from agentic_rag.ingestion import (
    clear_session_store,
    create_session_store,
    ingest_uploaded_documents,
)
from agentic_rag.main import build_agent, initial_state

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖")
st.title("🤖 Agentic RAG Document Assistant")
st.caption("Upload documents, then ask questions about their content.")

EXAMPLES = [
    "Summarize the main points of my document",
    "What are the key conclusions?",
    "Explain the main topic in simple terms",
]
BADGES = {
    "documents": "📄 Your documents",
    "web_search": "🌐 Web search",
    "direct": "💬 Direct reply",
    "insufficient_evidence": "⚠️ Not enough evidence",
}

# ---------------- Session setup ----------------
if "store" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.store = create_session_store()
    st.session_state.messages = []
    st.session_state.uploaded_names = []
    st.session_state.agent = None
    st.session_state.keys_sig = None


def friendly_error(e: Exception) -> str:
    t = str(e).lower()
    if any(w in t for w in ("401", "invalid api key", "invalid_api_key", "unauthorized", "authentication")):
        return "🔑 An API key was rejected. Please check your Groq / Tavily keys in the sidebar."
    if "429" in t or "rate limit" in t or "rate_limit" in t:
        return "⏳ Rate limit reached. Please wait a minute and try again."
    return f"Something went wrong: {e}"


def step_label(node: str, update: dict) -> str:
    update = update or {}
    if node == "route_question":
        if update.get("route") == "direct":
            return "💬 Simple message, replying directly"
        q = update.get("question", "")
        return f"🧭 Understanding your question → *{q}*" if q else "🧭 Understanding your question"
    if node == "retrieve_kb":
        return f"📚 Searched documents ({len(update.get('kb_docs', []))} passages found)"
    if node == "grade_kb_evidence":
        return ("✅ Document evidence looks good" if update.get("kb_grade") == "good"
                else "⚠️ Documents don't answer this well, trying the web")
    if node == "search_web":
        return ("🌐 Searched the web" if update.get("web_results")
                else "🌐 Web search unavailable (no Tavily key or no results)")
    if node == "grade_web_evidence":
        return "✅ Web evidence looks good" if update.get("web_grade") == "good" else "⚠️ Web evidence is weak"
    if node == "rewrite_query":
        return f"✏️ Rewrote the query → *{update.get('current_query', '')}*"
    if node in ("generate_from_kb", "generate_from_web", "direct_answer", "answer_insufficient"):
        return "✍️ Wrote the answer"
    return node


def render_sources(sources):
    if not sources:
        return
    with st.expander("📄 View document sources"):
        for s in sources:
            page = f" (page {s['page']})" if s.get("page") else ""
            st.markdown(f"**{s['source']}**{page}")
            st.caption(s.get("snippet", ""))


def render_footer(source: str, sources: list):
    st.caption(BADGES.get(source, f"Source: {source}"))
    render_sources(sources)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🔑 API keys")
    groq_key = st.text_input(
        "Groq API key", type="password", placeholder="gsk_...",
        help="Required. Free key at console.groq.com/keys",
    )
    tavily_key = st.text_input(
        "Tavily API key (optional)", type="password", placeholder="tvly-...",
        help="Enables web search when your documents aren't enough. Free key at app.tavily.com",
    )
    st.caption("Keys stay in your browser session only. They are never saved or logged.")

    # (Re)build the agent whenever the keys change; the document store is kept.
    sig = (groq_key, tavily_key)
    if st.session_state.keys_sig != sig:
        try:
            st.session_state.agent = build_agent(st.session_state.store, groq_key, tavily_key)
        except EnvironmentError:
            st.session_state.agent = None
        st.session_state.keys_sig = sig

    if st.session_state.agent is None:
        st.warning("Enter your Groq API key to start.")
    elif not tavily_key:
        st.caption("Web search is off unless a Tavily key is available.")

    st.divider()
    st.header("Your documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            try:
                with st.spinner("Processing documents..."):
                    count = ingest_uploaded_documents(uploaded_files, st.session_state.store)
                st.session_state.uploaded_names = sorted(
                    set(st.session_state.uploaded_names) | {f.name for f in uploaded_files}
                )
                st.success(f"Indexed {count} chunks!")
            except Exception as e:
                st.error(f"Processing failed: {e}")

    if st.session_state.uploaded_names:
        st.markdown("**Indexed in this session:**")
        for name in st.session_state.uploaded_names:
            st.write(f"- {name}")
        if st.button("Clear documents", use_container_width=True):
            clear_session_store(st.session_state.store)
            st.session_state.uploaded_names = []
            st.rerun()
    else:
        st.info("No documents uploaded yet. Upload a file and click Process Documents to start.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

has_docs = bool(st.session_state.uploaded_names)
has_agent = st.session_state.agent is not None
can_chat = has_docs and has_agent

# ---------------- Chat history ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_footer(message.get("source", "unknown"), message.get("sources", []))

# ---------------- Empty state ----------------
if not st.session_state.messages:
    if can_chat:
        st.markdown("#### Try asking")
        for i, example in enumerate(EXAMPLES):
            if st.button(example, key=f"example_{i}"):
                st.session_state.pending_question = example
                st.rerun()
    else:
        st.markdown("#### 👋 Get started")
        st.markdown(
            "1. Enter your **Groq API key** in the sidebar\n"
            "2. Upload a document and click **Process Documents**\n"
            "3. Ask questions!"
        )

# ---------------- New question ----------------
if not has_agent:
    placeholder = "Enter your Groq API key in the sidebar first"
elif not has_docs:
    placeholder = "Upload a document first to start chatting"
else:
    placeholder = "Ask a question about your documents..."

typed = st.chat_input(placeholder, disabled=not can_chat)
question = typed or (st.session_state.pop("pending_question", None) if can_chat else None)

if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-6:]
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    try:
        with st.chat_message("assistant"):
            result = initial_state(question, history)
            with st.status("Working on it...", expanded=True) as status:
                for chunk in st.session_state.agent.stream(result, stream_mode="updates"):
                    for node, update in chunk.items():
                        result.update(update or {})
                        status.write(step_label(node, update))
                status.update(label="Done", state="complete", expanded=False)

            answer = result.get("answer") or "No answer was returned."
            source = result.get("source_used", "unknown")
            sources = result.get("sources", [])

            st.write(answer)
            render_footer(source, sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "source": source, "sources": sources}
        )

    except Exception as e:
        st.error(friendly_error(e))