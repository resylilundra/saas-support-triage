import json
from types import SimpleNamespace

import pytest

import src.llm_client as llm_client


VALID_FIELDS = {
    "language": "fr",
    "issue_type": "bug",
    "priority": "normal",
    "product_area": "notifications",
    "summary": "The user receives duplicate task notifications.",
}


class FakeCompletions:
    def __init__(self, payload=None, error=None, usage=None):
        self.payload = payload
        self.error = error
        self.usage = usage or SimpleNamespace(prompt_tokens=120, completion_tokens=40)
        self.request = None

    def create(self, **request):
        self.request = request
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.payload)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=self.usage)


def make_client_factory(completions):
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def factory(**kwargs):
        assert set(kwargs) == {"api_key"}
        return client

    return factory


def test_extractor_sends_prompt_and_returns_usage(monkeypatch):
    completions = FakeCompletions(payload=json.dumps(VALID_FIELDS))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    extractor = llm_client.LLMExtractor(
        model="test-model", client_factory=make_client_factory(completions)
    )

    result = extractor.extract("Notifications en double", "Je reçois deux notifications.")

    assert json.loads(result.payload) == VALID_FIELDS
    assert (result.prompt_tokens, result.completion_tokens) == (120, 40)
    assert completions.request["model"] == "test-model"
    assert completions.request["temperature"] == 0
    assert completions.request["response_format"] == {"type": "json_object"}
    messages = completions.request["messages"]
    assert "Nimbus" in messages[0]["content"]
    assert "Notifications en double" in messages[1]["content"]


def test_extractor_includes_validation_feedback(monkeypatch):
    completions = FakeCompletions(payload=json.dumps(VALID_FIELDS))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    extractor = llm_client.LLMExtractor(client_factory=make_client_factory(completions))

    extractor.extract("s", "b", feedback="bad enum value")

    sent_content = completions.request["messages"][1]["content"]
    assert "bad enum value" in sent_content
    assert "corrected JSON" in sent_content


def test_extractor_reuses_client(monkeypatch):
    completions = FakeCompletions(payload=json.dumps(VALID_FIELDS))
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    extractor = llm_client.LLMExtractor(client_factory=factory)
    extractor.extract("s", "b")
    extractor.extract("s", "b")

    assert len(calls) == 1


def test_extractor_needs_api_key(monkeypatch):
    completions = FakeCompletions(payload=json.dumps(VALID_FIELDS))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    extractor = llm_client.LLMExtractor(client_factory=make_client_factory(completions))

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        extractor.extract("s", "b")


def test_extractor_rejects_empty_response(monkeypatch):
    completions = FakeCompletions(payload="")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    extractor = llm_client.LLMExtractor(client_factory=make_client_factory(completions))

    with pytest.raises(ValueError, match="empty response"):
        extractor.extract("s", "b")


def test_extractor_does_not_hide_api_errors(monkeypatch):
    completions = FakeCompletions(error=RuntimeError("API unavailable"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    extractor = llm_client.LLMExtractor(client_factory=make_client_factory(completions))

    with pytest.raises(RuntimeError, match="API unavailable"):
        extractor.extract("s", "b")
