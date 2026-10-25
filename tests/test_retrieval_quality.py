from src import config
from src.ingestion import store
from src.ingestion.store import KnowledgeBase, bm25_scores, rrf
from tests.test_pipeline_e2e import FakeEmbeddings


def test_bm25_prefers_exact_terms():
    corpus = ["python and pandas for data", "gardening with roses", "kubernetes deployment on aws"]
    s = bm25_scores("kubernetes aws", corpus, corpus)
    assert s[2] > s[0] and s[2] > s[1]


def test_rrf_fuses_orderings():
    assert rrf([0, 1, 2], [2, 1, 0])[0] in (0, 2)
    assert rrf([0, 1, 2], [0, 2, 1])[0] == 0


def _kb():
    from src.ingestion.loaders import RawDoc

    kb = KnowledgeBase(embeddings=FakeEmbeddings())
    kb.add([RawDoc(source="cv", text="Built a Kubernetes platform on AWS.\n\nLed a gardening club.\n\nWrote Python ETL jobs.")])
    return kb


def test_search_unchanged_by_default():
    kb = _kb()
    assert len(kb.search("kubernetes", k=2)) <= 2


def test_hybrid_and_reranker_reorder(monkeypatch):
    kb = _kb()
    monkeypatch.setattr(config, "HYBRID_SEARCH", True)
    monkeypatch.setattr(config, "RERANKER_MODEL", "fake-ce")

    class FakeCE:
        def predict(self, pairs):
            return [1.0 if "gardening" in t else 0.0 for _, t in pairs]

    monkeypatch.setattr(store, "get_reranker", lambda name: FakeCE())
    top = kb.search("kubernetes", k=1)
    assert top and "gardening" in top[0][0]
