from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="Use kb for questions needing the knowledge base; direct for greetings/simple chat."
    )


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="good means evidence can answer the question; weak means evidence is insufficient."
    )


class AgentState(TypedDict, total=False):
    question: str
    current_query: str
    route: str          # router decision: "kb" | "direct"
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str    # where the answer came from: documents | web_search | direct | ...
    retry_count: int
    chat_history: List[dict]  # recent turns: [{"role", "content"}]
    sources: List[dict]  # citations: [{"source", "page", "snippet"}]