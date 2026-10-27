from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes import make_nodes


def build_graph(llm, retriever, web_search):
    nodes = make_nodes(llm, retriever, web_search)
    workflow = StateGraph(AgentState)

    for node_name in (
        "route_question",
        "retrieve_kb",
        "grade_kb_evidence",
        "search_web",
        "grade_web_evidence",
        "rewrite_query",
        "generate_from_kb",
        "generate_from_web",
        "direct_answer",
        "answer_insufficient",
    ):
        workflow.add_node(node_name, nodes[node_name])

    workflow.add_edge(START, "route_question")
    workflow.add_conditional_edges(
        "route_question",
        nodes["route_after_router"],
        {"retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"},
    )
    workflow.add_edge("retrieve_kb", "grade_kb_evidence")
    workflow.add_conditional_edges(
        "grade_kb_evidence",
        nodes["decide_after_kb_grade"],
        {"generate_from_kb": "generate_from_kb", "search_web": "search_web"},
    )
    workflow.add_edge("search_web", "grade_web_evidence")
    workflow.add_conditional_edges(
        "grade_web_evidence",
        nodes["decide_after_web_grade"],
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "answer_insufficient": "answer_insufficient",
        },
    )
    # Rewritten queries retry the private KB first, then fall back to web again if needed.
    workflow.add_edge("rewrite_query", "retrieve_kb")
    workflow.add_edge("generate_from_kb", END)
    workflow.add_edge("generate_from_web", END)
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("answer_insufficient", END)

    return workflow.compile()
