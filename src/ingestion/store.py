"""Personal knowledge base: chunk + embed everything the candidate gives us."""
from __future__ import annotations

import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config
from ..llm import get_embeddings
from .loaders import RawDoc


class KnowledgeBase:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or get_embeddings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        self._store = Chroma(
            collection_name=f"kb_{uuid.uuid4().hex[:12]}",
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.n_chunks = 0

    def add(self, docs: list[RawDoc]) -> int:
        chunks: list[Document] = []
        for d in docs:
            if not d.text.strip():
                continue
            for i, piece in enumerate(self._splitter.split_text(d.text)):
                chunks.append(Document(page_content=piece, metadata={"source": d.source, "i": i}))
        if chunks:
            self._store.add_documents(chunks)
            self.n_chunks += len(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        """Returns (text, source, relevance) triples; relevance is cosine similarity."""
        if self.n_chunks == 0:
            return []
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        return [(doc.page_content, doc.metadata.get("source", "?"), float(s)) for doc, s in results]
