import hashlib
import io
import uuid
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    SOURCE_URL,
    create_embeddings,
)


@lru_cache(maxsize=1)
def _embeddings():
    """Load the embedding model once per process."""
    return create_embeddings()


def _splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, add_start_index=True
    )


# --------------------------------------------------------------------------
# Default (persistent) knowledge base: the public docs page
# --------------------------------------------------------------------------
def create_retriever(k: int = 3):
    """Persistent store for the default source URL (shared, read-only)."""
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=_embeddings(),
    )

    if vectorstore._collection.count() == 0:
        print(f"[Ingestion] Loading source: {SOURCE_URL}")
        loader = WebBaseLoader(
            web_path=(SOURCE_URL,),
            requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0 Agentic-RAG"}},
        )
        raw_docs = loader.load()
        if not raw_docs:
            raise RuntimeError(f"No documents were loaded from {SOURCE_URL}")

        chunks = _splitter().split_documents(raw_docs)
        if not chunks:
            raise RuntimeError("Document splitting produced no chunks.")

        ids = [f"{doc.metadata.get('source', SOURCE_URL)}:{i}" for i, doc in enumerate(chunks)]
        vectorstore.add_documents(documents=chunks, ids=ids)
        print(f"[Ingestion] Stored {len(chunks)} chunks.")
    else:
        print(f"[Ingestion] Reusing {vectorstore._collection.count()} stored chunks.")

    return vectorstore.as_retriever(search_kwargs={"k": k})


# --------------------------------------------------------------------------
# Per-session uploads (in-memory, never written to disk)
# --------------------------------------------------------------------------
def create_session_store() -> Chroma:
    """
    Fresh, empty, in-memory vector store for ONE user session.
    No persist_directory => nothing survives a page refresh / restart.
    Unique collection name => sessions never see each other's files.
    """
    return Chroma(
        collection_name=f"uploads_{uuid.uuid4().hex}",
        embedding_function=_embeddings(),
    )


def clear_session_store(store: Chroma) -> None:
    """Remove every uploaded chunk from this session's store."""
    ids = store.get().get("ids", [])
    if ids:
        store.delete(ids=ids)


def _file_to_documents(name: str, data: bytes) -> list[Document]:
    lower = name.lower()
    if lower.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": name})] if text.strip() else []

    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        docs = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                # One Document per page so citations can include page numbers.
                docs.append(
                    Document(page_content=text, metadata={"source": name, "page": page_no})
                )
        return docs

    raise ValueError(f"Unsupported file type: {name}")


def ingest_uploaded_documents(files, store: Chroma) -> int:
    """Add uploaded files to this session's store. Returns number of new chunks."""
    documents: list[Document] = []
    for file in files:
        # Streamlit's UploadedFile.read() returns b"" on reruns; getvalue() is safe.
        data = file.getvalue() if hasattr(file, "getvalue") else file.read()
        documents.extend(_file_to_documents(file.name, data))

    if not documents:
        raise ValueError("No readable text found in uploaded files.")

    chunks = _splitter().split_documents(documents)

    unique = {}
    for chunk in chunks:
        key = f"{chunk.metadata['source']}:{chunk.metadata.get('page', 0)}:{chunk.page_content}"
        unique[hashlib.sha256(key.encode("utf-8")).hexdigest()] = chunk

    store.add_documents(documents=list(unique.values()), ids=list(unique.keys()))
    return len(unique)


class SessionRetriever:
    """
    Searches ONLY this session's uploaded documents.
    Returns [] when nothing is uploaded (the graph then treats evidence as weak).
    Optionally pass a base_retriever with include_base=True to also search a shared KB.
    """

    def __init__(self, session_store: Chroma, base_retriever=None, k: int = 3, include_base: bool = False):
        self.session_store = session_store
        self.base_retriever = base_retriever
        self.k = k
        self.include_base = include_base and base_retriever is not None

    def has_uploads(self) -> bool:
        return self.session_store._collection.count() > 0

    def invoke(self, query: str, *args, **kwargs):
        if not self.has_uploads():
            return []
        docs = self.session_store.similarity_search(query, k=self.k)
        if self.include_base:
            docs = docs + self.base_retriever.invoke(query)
        return docs