import pytest
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src import llm as llm_module
from src.llm import ModelChain, _parse_models, get_llm, invoke_structured
from src.schemas import JobProfile

OK = JobProfile(title="Dev", requirements=[], keywords=[])
PROMPT = ChatPromptTemplate.from_messages([("human", "{x}")])


class FakeChat(RunnableLambda):
    """A chat model that answers with its own name, or fails like an overloaded provider."""

    def __init__(self, name, fails=False):
        def reply(_):
            if fails:
                raise RuntimeError(f"{name}: 503 UNAVAILABLE")
            return AIMessage(content=name)

        super().__init__(reply)
        self.name, self.fails, self.model_name = name, fails, name

    def with_structured_output(self, schema, **kw):
        def structured(_):
            if self.fails:
                raise RuntimeError(f"{self.name}: 503 UNAVAILABLE")
            return OK.model_copy(update={"title": self.name})

        return RunnableLambda(structured)


def chain(*models):
    return ModelChain(runnable=models[0], fallbacks=list(models[1:]), exceptions_to_handle=(Exception,))


def test_parse_models_keeps_order_and_default_provider():
    assert _parse_models("a-1, b-2 ,c-3", "google") == [("google", "a-1"), ("google", "b-2"), ("google", "c-3")]


def test_parse_models_accepts_a_provider_prefix_per_entry():
    assert _parse_models("google:gemini-x,groq:openai/gpt-oss-120b", "google") == [
        ("google", "gemini-x"),
        ("groq", "openai/gpt-oss-120b"),
    ]


def test_parse_models_does_not_mistake_ollama_tags_for_providers():
    assert _parse_models("qwen2.5:7b", "ollama") == [("ollama", "qwen2.5:7b")]


def test_next_model_answers_when_the_first_is_unavailable():
    c = chain(FakeChat("first", fails=True), FakeChat("second", fails=True), FakeChat("third"))
    assert (c | StrOutputParser()).invoke("hi") == "third"


def test_structured_output_also_falls_through_in_order():
    c = chain(FakeChat("first", fails=True), FakeChat("second"), FakeChat("third"))
    assert invoke_structured(PROMPT, JobProfile, {"x": "hi"}, c, attempts=1).title == "second"


def test_the_first_model_is_used_when_it_works():
    c = chain(FakeChat("first"), FakeChat("second"))
    assert (c | StrOutputParser()).invoke("hi") == "first"


def test_get_llm_builds_a_chain_from_a_list(monkeypatch):
    built = []
    monkeypatch.setattr(
        llm_module, "_build_llm", lambda p, m, t: built.append((p, m)) or FakeChat(m)
    )
    model = get_llm(0.2, provider="google", model="gemini-3.8-flash, gemini-3.6-flash")
    assert isinstance(model, ModelChain)
    assert built == [("google", "gemini-3.8-flash"), ("google", "gemini-3.6-flash")]
    assert model.model_name == "gemini-3.8-flash > gemini-3.6-flash"


def test_get_llm_returns_a_plain_model_for_a_single_name(monkeypatch):
    monkeypatch.setattr(llm_module, "_build_llm", lambda p, m, t: FakeChat(m))
    assert not isinstance(get_llm(0.2, provider="google", model="gemini-3.8-flash"), ModelChain)


def test_empty_model_is_an_error(monkeypatch):
    monkeypatch.setattr(llm_module.config, "LLM_MODEL", " , ")
    with pytest.raises(ValueError, match="No model"):
        get_llm()
