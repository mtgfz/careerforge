import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src import llm as llm_module
from src.llm import invoke_structured
from src.schemas import JobProfile

PROMPT = ChatPromptTemplate.from_messages([("human", "{x}")])
OK = JobProfile(title="Dev", requirements=[], keywords=[])


class FlakyLLM:
    """with_structured_output() returns a runnable that fails `failures` times, then succeeds."""

    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def with_structured_output(self, schema, **kw):
        def run(_):
            self.calls += 1
            if self.calls <= self.failures:
                raise ValueError("tool_use_failed")
            return OK

        return RunnableLambda(run)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)


def test_retries_until_success():
    model = FlakyLLM(failures=2)
    assert invoke_structured(PROMPT, JobProfile, {"x": "hi"}, model, attempts=3) == OK
    assert model.calls == 3


def test_fallback_used_after_primary_exhausted():
    primary, backup = FlakyLLM(failures=99), FlakyLLM(failures=0)
    assert invoke_structured(PROMPT, JobProfile, {"x": "hi"}, primary, fallback=backup, attempts=2) == OK
    assert primary.calls == 2 and backup.calls == 1


def test_raises_last_error_when_everything_fails():
    primary, backup = FlakyLLM(failures=99), FlakyLLM(failures=99)
    with pytest.raises(ValueError, match="tool_use_failed"):
        invoke_structured(PROMPT, JobProfile, {"x": "hi"}, primary, fallback=backup, attempts=2)
    assert primary.calls == 2 and backup.calls == 2


class RateLimited(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 429


class QuotaLLM:
    model_name = "openai/gpt-oss-120b"

    def __init__(self, message):
        self.calls = 0
        self.message = message

    def with_structured_output(self, schema, **kw):
        def run(_):
            self.calls += 1
            raise RateLimited(self.message)

        return RunnableLambda(run)


DAILY = "Rate limit reached on tokens per day (TPD): Limit 200000. Please try again in 33m30.096s."


def test_daily_quota_stops_retrying_and_explains():
    model = QuotaLLM(DAILY)
    with pytest.raises(llm_module.QuotaExceeded, match="daily quota of 'openai/gpt-oss-120b'.*33m30"):
        invoke_structured(PROMPT, JobProfile, {"x": "hi"}, model, attempts=3)
    assert model.calls == 1


def test_daily_quota_falls_back_to_the_other_model():
    primary, backup = QuotaLLM(DAILY), FlakyLLM(failures=0)
    assert invoke_structured(PROMPT, JobProfile, {"x": "hi"}, primary, fallback=backup) == OK
    assert primary.calls == 1 and backup.calls == 1


def test_per_minute_limit_waits_then_retries(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_module.time, "sleep", waits.append)
    model = QuotaLLM("Rate limit reached on tokens per minute (TPM). Please try again in 7.5s.")
    with pytest.raises(llm_module.QuotaExceeded):
        invoke_structured(PROMPT, JobProfile, {"x": "hi"}, model, attempts=2)
    assert model.calls == 2 and waits == [7.5]


def test_quota_error_ignores_other_errors():
    assert llm_module.quota_error(ValueError("boom")) is None


class GoogleStyleError(Exception):
    """google-genai errors carry the HTTP status in `code`, not `status_code`."""

    code = 429


def test_google_per_minute_limit_is_understood():
    err = GoogleStyleError("429 RESOURCE_EXHAUSTED. Quota exceeded. Please retry in 12.5s.")
    q = llm_module.quota_error(err)
    assert q and not q.daily and q.wait == 12.5


def test_google_daily_limit_is_understood():
    err = GoogleStyleError(
        "RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded"
    )
    q = llm_module.quota_error(err)
    assert q and q.daily


def test_quota_error_follows_the_cause_chain():
    try:
        try:
            raise GoogleStyleError("RESOURCE_EXHAUSTED retry in 3s")
        except GoogleStyleError as inner:
            raise RuntimeError("wrapped by the integration") from inner
    except RuntimeError as outer:
        q = llm_module.quota_error(outer)
    assert q and q.wait == 3.0
