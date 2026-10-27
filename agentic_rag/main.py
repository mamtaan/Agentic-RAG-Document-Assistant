from .config import create_llm, create_web_search
from .graph import build_graph
from .ingestion import SessionRetriever, create_session_store


def initial_state(question: str, history: list | None = None) -> dict:
    return {
        "question": question,
        "current_query": question,
        "route": "",
        "chat_history": history or [],
        "retry_count": 0,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "sources": [],  # always reset so old citations never leak into new answers
    }


def build_agent(session_store, groq_key=None, tavily_key=None, k: int = 3):
    """Compile the graph for one session with the user's own API keys.
    Raises EnvironmentError if no Groq key is available. Tavily is optional."""
    retriever = SessionRetriever(session_store, k=k)
    return build_graph(
        llm=create_llm(groq_key),
        retriever=retriever,
        web_search=create_web_search(tavily_key),  # may be None -> web search disabled
    )


def build_app_for_session(k: int = 3):
    """CLI helper: uses keys from the environment / .env."""
    session_store = create_session_store()
    return session_store, build_agent(session_store, k=k)


def format_sources(sources: list[dict]) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        page = f", p.{s['page']}" if s.get("page") else ""
        lines.append(f"  [{i}] {s['source']}{page}")
    return "\n".join(lines)


def main():
    _, app = build_app_for_session()
    print("Agentic RAG graph compiled.")
    try:  # IPython is optional
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception as e:
        print("Graph visualization skipped:", e)

    print("\nType 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            result = app.invoke(initial_state(question, history))
            print(f"\nAssistant: {result.get('answer', 'No answer was returned.')}")
            print(f"Source: {result.get('source_used', 'unknown')}")
            if result.get("sources"):
                print("Cited documents:\n" + format_sources(result["sources"]))
            print()
            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.get("answer", "")},
            ]
        except Exception as e:
            print(f"\nAn error occurred while processing the question: {e}\n")


if __name__ == "__main__":
    main()