"""Personal knowledge base: chunk + embed everything the candidate gives us."""
from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config
from ..llm import get_embeddings
from .loaders import RawDoc


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_scores(query: str, docs: list[str], corpus: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 of `query` against `docs`, with IDF and average length taken from the whole `corpus`."""
    n = max(len(corpus), 1)
    toks = [_tokens(c) for c in corpus]
    avg = sum(len(t) for t in toks) / n or 1.0
    df = Counter(w for t in toks for w in set(t))
    q = set(_tokens(query))
    out = []
    for d in docs:
        dt = _tokens(d)
        tf = Counter(dt)
        score = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
            score += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(dt) / avg))
        out.append(score)
    return out


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    """Reciprocal rank fusion of several orderings of the same item indices."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: -score[i])


@lru_cache(maxsize=2)
def get_reranker(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)


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
        self._texts: list[str] = []

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
            self._texts.extend(c.page_content for c in chunks)
        return len(chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[str, str, float]]:
        """Returns (text, source, relevance) triples; relevance is the dense cosine similarity.

        With HYBRID_SEARCH and/or RERANKER_MODEL, a larger dense pool is re-ordered before taking the top k.
        """
        if self.n_chunks == 0:
            return []
        rerank = config.HYBRID_SEARCH or bool(config.RERANKER_MODEL)
        pool = max(k, config.RERANK_POOL) if rerank else k
        results = self._store.similarity_search_with_relevance_scores(query, k=pool)
        hits = [(doc.page_content, doc.metadata.get("source", "?"), float(s)) for doc, s in results]
        if rerank and len(hits) > 1:
            texts = [h[0] for h in hits]
            order = list(range(len(hits)))  # dense order
            if config.HYBRID_SEARCH:
                lex = bm25_scores(query, texts, self._texts)
                bm_order = sorted(order, key=lambda i: -lex[i])
                order = rrf(order, bm_order)
            if config.RERANKER_MODEL:
                ce = get_reranker(config.RERANKER_MODEL)
                ce_scores = ce.predict([(query, texts[i]) for i in order])
                order = [i for _, i in sorted(zip(ce_scores, order), key=lambda x: -x[0])]
            hits = [hits[i] for i in order]
        return hits[:k]
