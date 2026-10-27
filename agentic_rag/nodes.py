from typing import Literal
from langchain_core.documents import Document

from .config import MAX_RETRIES
from .state import AgentState, EvidenceGrade, RouteDecision


def make_nodes(llm, retriever, web_search):
    """Build graph node functions with their dependencies injected."""
    router_llm = llm.with_structured_output(RouteDecision, method="json_mode")
    grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")

    
    def format_history(history, limit=6, max_chars=500):
        return "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:max_chars]}"
            for m in history[-limit:]
        )

    def condense_question(question, history):
        """Turn a follow-up like 'what about the second one?' into a standalone question."""
        try:
            rewritten = llm.invoke(f"""
Given the conversation below, rewrite the user's latest message as a single
standalone question that can be understood without the conversation.
If it is already standalone, return it unchanged. Do not answer it.
Return only the question.

Conversation:
{format_history(history)}

Latest message: {question}
""").content.strip()
            return rewritten or question
        except Exception as e:
            print("[Condense] failed, using original question:", e)
            return question

    def route_question(state: AgentState):
        question = state["question"]

        try:
            decision = router_llm.invoke(f"""
You are routing questions for a document assistant.

Route to "kb" for any question that may require information
from the user's uploaded documents or knowledge base.

Route to "direct" ONLY for greetings, thanks, or simple
conversation that does not require factual research.

If unsure, choose "kb".

Question: {question}

Return valid JSON like {{"route": "kb"}}.
""")
            route = decision.route
        except Exception as e:
            print("[Router] parse failure, defaulting to kb:", e)
            route = "kb"

        print("[Router]", route)
        history = state.get("chat_history", [])
        if route == "kb" and history:
            question = condense_question(question, history)
            print("[Condense]", question)
        return {"question": question, "current_query": question, "route": route}


    def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
        return "retrieve_kb" if state.get("route") == "kb" else "direct_answer"

    def retrieve_kb(state: AgentState):
        query = state.get("current_query", state["question"])
        docs = retriever.invoke(query)
        print(f"[KB Retriever] Query: {query}")
        print(f"[KB Retriever] Retrieved: {len(docs)} chunks")
        return {"kb_docs": docs}

    def grade_kb_evidence(state: AgentState):
        if not state.get("kb_docs"):
            print("[KB Grader] no docs -> weak")
            return {"kb_grade": "weak"}
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
            for doc in state.get("kb_docs", [])
        )
        grade = grader_llm.invoke(f"""
You are an evidence grader.
Question: {state['question']}
Private KB evidence:
{context}
Can this evidence answer the question? Return "good" if it can answer,
otherwise return "weak". Return valid JSON like {{"grade": "good"}}.
""")
        print("[KB Grader]", grade.grade)
        return {"kb_grade": grade.grade}

    def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
        return "generate_from_kb" if state.get("kb_grade") == "good" else "search_web"

    def search_web_node(state: AgentState):
        if web_search is None:
            print("[Tavily Search] disabled (no API key)")
            return {"web_results": "", "source_used": "web"}
        query = state.get("current_query", state["question"])
        print(f"[Tavily Search] Query: {query}")
        result = web_search.invoke({"query": query})
        lines = []
        if isinstance(result, dict):
            answer = result.get("answer", "")
            if answer:
                lines.append(f"Tavily answer: {answer}")
            # Tavily package versions may use either `results` or `result`.
            items = result.get("results", result.get("result", []))
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        lines.append(str(item))
                        continue
                    title = item.get("title", "")
                    url = item.get("url", "")
                    content = item.get("content", "")
                    lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            web_text = "\n\n".join(lines) if lines else str(result)
        else:
            web_text = str(result)

        print("[Tavily Search] Result characters:", len(web_text))
        return {"web_results": web_text, "source_used": "web"}

    def grade_web_evidence(state: AgentState):
        if not state.get("web_results", "").strip():
            print("[Web Grader] no web results -> weak")
            return {"web_grade": "weak"}
        grade = grader_llm.invoke(f"""
You are an evidence grader.
Question: {state['question']}
Web search evidence:
{state.get('web_results', '')}
Can this evidence answer the question? Return "good" if it can answer,
otherwise return "weak". Return valid JSON like {{"grade": "good"}}.
""")
        print("[Web Grader]", grade.grade)
        return {"web_grade": grade.grade}

    def decide_after_web_grade(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
        if state.get("web_grade") == "good":
            return "generate_from_web"
        if web_search is None:  # retrying is pointless without web search
            return "answer_insufficient"
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "rewrite_query"
        return "answer_insufficient"

    def rewrite_query(state: AgentState):
        retry_count = state.get("retry_count", 0) + 1
        rewritten = llm.invoke(f"""
Rewrite the question for better retrieval and web search.
Preserve the original intent, make it specific and search-friendly, and do not answer.
Return only the rewritten query.
Original question: {state['question']}
""").content.strip()
        if not rewritten:
            rewritten = state["question"]
        print("[Rewriter]", rewritten)
        return {"current_query": rewritten, "retry_count": retry_count}

    def generate_from_kb(state: AgentState):
        docs = state.get("kb_docs", [])

        def label(doc):
            name = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")
            return f"{name}, p.{page}" if page else name

        context = "\n\n".join(
            f"[{i}] ({label(doc)})\n{doc.page_content}"
            for i, doc in enumerate(docs, start=1)
        )

        answer = llm.invoke(f"""
Answer the user's question using only the provided documents.
Cite the supporting documents inline using their numbers, like [1] or [2][3].
If the documents don't contain the answer, say so. Do not invent facts.

Documents:
{context}

Question: {state['question']}
""")

        sources, seen = [], set()
        for doc in docs:
            key = (doc.metadata.get("source", "Unknown"), doc.metadata.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "source": key[0],
                "page": key[1],
                "snippet": doc.page_content[:300].strip(),
            })

        return {"answer": answer.content, "source_used": "documents", "sources": sources}

    def generate_from_web(state: AgentState):
        answer = llm.invoke(f"""
You are a technical instructor. The private KB was insufficient, so web search was used.
Answer using ONLY the web search context. Be beginner-friendly, do not invent unsupported
facts, mention Tavily web search, include source type: Web Search, and include useful URLs
when present in the context.
Question: {state['question']}
Web search context:
{state.get('web_results', '')}
""").content
        return {"answer": answer, "source_used": "web_search"}

    def direct_answer(state: AgentState):
        history = format_history(state.get("chat_history", []))
        answer = llm.invoke(
            f"Conversation so far:\n{history}\n\n"
            f"Respond briefly and naturally to this message: {state['question']}"
        ).content
        return {"answer": answer, "source_used": "direct"}

    def answer_insufficient(state: AgentState):
        return {
            "answer": (
                "I could not find enough reliable evidence in the private knowledge base "
                "or web search results to answer confidently. Please provide more specific "
                "documents or rephrase the question."
            ),
            "source_used": "insufficient_evidence",
        }

    return {
        "route_question": route_question,
        "route_after_router": route_after_router,
        "retrieve_kb": retrieve_kb,
        "grade_kb_evidence": grade_kb_evidence,
        "decide_after_kb_grade": decide_after_kb_grade,
        "search_web": search_web_node,
        "grade_web_evidence": grade_web_evidence,
        "decide_after_web_grade": decide_after_web_grade,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "answer_insufficient": answer_insufficient,
    }