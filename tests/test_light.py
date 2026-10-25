import pytest

pytest.importorskip("langchain_chroma")

from src import config
from src import llm as llm_module
from src import pipeline as pipeline_module
from src.ingestion.loaders import load_text
from src.ingestion.store import KnowledgeBase
from src.pipeline import run_pipeline
from src.schemas import Candidate
from tests.test_pipeline_e2e import FakeEmbeddings, FakeLLM


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_MODEL", "big")
    monkeypatch.setattr(llm_module, "get_llm", lambda t=0.2, provider=None, model=None: (provider, model))


def test_light_model_is_used_when_configured(fake_models, monkeypatch):
    monkeypatch.setattr(config, "LIGHT_PROVIDER", "groq")
    monkeypatch.setattr(config, "LIGHT_MODEL", "small")
    assert llm_module.get_light_llm() == ("groq", "small")
    assert llm_module.light_is_distinct()


def test_light_model_defaults_to_the_main_one(fake_models, monkeypatch):
    monkeypatch.setattr(config, "LIGHT_PROVIDER", "")
    monkeypatch.setattr(config, "LIGHT_MODEL", "")
    assert llm_module.get_light_llm() == ("groq", "big")
    assert not llm_module.light_is_distinct()


def test_light_provider_needs_its_own_model(fake_models, monkeypatch):
    monkeypatch.setattr(config, "LIGHT_PROVIDER", "openai")
    monkeypatch.setattr(config, "LIGHT_MODEL", "")
    with pytest.raises(ValueError, match="LIGHT_MODEL"):
        llm_module.get_light_llm()


def test_failed_extras_do_not_lose_the_cv(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("quota gone")

    monkeypatch.setattr(pipeline_module, "cover_letter", boom)
    emb = FakeEmbeddings()
    kb = KnowledgeBase(embeddings=emb)
    kb.add([load_text("I built internal tools in Python with FastAPI and Docker.", "cv.md")])

    res = run_pipeline(kb, "ML intern, Python and Docker", Candidate(name="T"), llm=FakeLLM(), embeddings=emb)

    assert res.cover_letter == "(Not generated: quota gone)"
    assert res.questions  # the other bonus output still works
    assert res.cv.projects


def test_google_provider_builds_a_gemini_chat_model(monkeypatch):
    pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    model = llm_module.get_llm(0.1, provider="google", model="gemini-3.8-flash")
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert "gemini-3.8-flash" in model.model
