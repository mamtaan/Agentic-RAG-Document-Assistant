import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

SOURCE_URL = os.getenv(
    "SOURCE_URL",
    "https://docs.langchain.com/oss/python/langgraph/agentic-rag",
)
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "agentic_rag_docs")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))


# Set ALLOW_SERVER_KEYS=false on a public deployment so visitors must bring their own keys.
ALLOW_SERVER_KEYS = os.getenv("ALLOW_SERVER_KEYS", "true").lower() == "true"


def _resolve_key(user_key, env_name):
    """User-provided key wins; the server's .env key is used only if allowed."""
    if user_key and user_key.strip():
        return user_key.strip()
    if ALLOW_SERVER_KEYS:
        return os.getenv(env_name) or None
    return None


def create_llm(api_key=None) -> ChatGroq:
    key = _resolve_key(api_key, "GROQ_API_KEY")
    if not key:
        raise EnvironmentError("Groq API key is missing. Enter it in the sidebar.")
    return ChatGroq(model=GROQ_MODEL, temperature=0, groq_api_key=key)


def create_web_search(api_key=None):
    """Returns None if no key is available (web search is optional)."""
    key = _resolve_key(api_key, "TAVILY_API_KEY")
    if not key:
        return None
    return TavilySearch(
        tavily_api_key=key,
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=False,
    )

def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


