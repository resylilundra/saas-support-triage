import json

import pytest

from src.llm_client import Extraction
from src.triage import TriagePipeline


class FakeExtractor:
    def __init__(self, fields, *, prompt_tokens=100, completion_tokens=20):
        self.payload = json.dumps(fields)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls = []

    def extract(self, subject, body, *, feedback=None):
        self.calls.append((subject, body, feedback))
        return Extraction(
            payload=self.payload,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


def fields(**overrides):
    base = {
        "language": "en",
        "issue_type": "bug",
        "priority": "normal",
        "product_area": "web_app",
        "summary": "A generic issue was reported.",
    }
    return {**base, **overrides}


def ticket(**overrides):
    base = {
        "id": "tx01",
        "subject": "Service down for everyone",
        "body": "Our whole team cannot open Nimbus.",
    }
    return {**base, **overrides}


def test_deterministic_escalation_cannot_be_overridden_by_model():
    extractor = FakeExtractor(fields(priority="low"))
    pipeline = TriagePipeline(extractor)

    result = pipeline.triage(ticket())

    assert result.priority == "low"
    assert result.escalation_required is True
    assert result.send_to_oncall is True
    assert result.escalation_rule == "service_wide_outage"
    assert (result.prompt_tokens, result.completion_tokens) == (100, 20)


def test_non_incident_is_not_sent_to_oncall():
    pipeline = TriagePipeline(FakeExtractor(fields()))
    result = pipeline.triage(
        ticket(subject="Can't sign in", body="Only my account is affected.")
    )

    assert result.escalation_required is False
    assert result.send_to_oncall is False
    assert result.escalation_rule is None


def test_result_as_dict_includes_all_scored_fields():
    pipeline = TriagePipeline(FakeExtractor(fields(language="es")))

    result = pipeline.triage(ticket()).as_dict()

    assert result["id"] == "tx01"
    assert result["language"] == "es"
    assert set(result) >= {
        "language",
        "issue_type",
        "priority",
        "product_area",
        "escalation_required",
    }


def test_invalid_model_payload_stops_the_pipeline():
    extractor = FakeExtractor(fields())
    extractor.payload = "not JSON"
    pipeline = TriagePipeline(extractor)

    with pytest.raises(ValueError, match="not valid JSON"):
        pipeline.triage(ticket())


def test_retry_with_validation_feedback_counts_all_tokens():
    class FlakyExtractor:
        def __init__(self):
            self.feedback_seen = []

        def extract(self, subject, body, *, feedback=None):
            self.feedback_seen.append(feedback)
            payload = (
                json.dumps(fields(issue_type="web_app"))
                if feedback is None
                else json.dumps(fields())
            )
            return Extraction(payload=payload, prompt_tokens=100, completion_tokens=20)

    extractor = FlakyExtractor()
    pipeline = TriagePipeline(extractor)

    result = pipeline.triage(ticket())

    assert result.issue_type == "bug"
    assert extractor.feedback_seen[0] is None
    assert "does not match schema" in extractor.feedback_seen[1]
    assert (result.prompt_tokens, result.completion_tokens) == (200, 40)


def test_retries_are_bounded():
    class AlwaysInvalidExtractor:
        def __init__(self):
            self.calls = 0

        def extract(self, subject, body, *, feedback=None):
            self.calls += 1
            return Extraction(payload="not JSON", prompt_tokens=1, completion_tokens=1)

    extractor = AlwaysInvalidExtractor()
    pipeline = TriagePipeline(extractor, max_validation_retries=2)

    with pytest.raises(ValueError, match="not valid JSON"):
        pipeline.triage(ticket())

    assert extractor.calls == 3
